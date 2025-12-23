import copy
import logging
import sys
import yaml
import os
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parallel import DataParallel

from src.dataset.masks.all_masks  import MutiBlockMaskCollector as MBMaskCollator
from src.help.utils  import apply_masks
from src.help.logging import (
    CSVLogger,
    gpu_timer,
    grad_logger,
    AverageMeter)
from src.help.utils import repeat_interleave_batch
from src.dataset.data.imagenet import make_imagenet1k, make_imagenet1k_fraction, make_imagenet1k_balanced_subset

from src.help.schedulers import (
    load_checkpoint,
    init_model,
    init_opt)
from src.dataset.data.transform import make_transforms

# --
log_timings = True
log_freq = 10  # Log every 10 iterations within an epoch
epoch_log_freq = 10  # Log detailed stats every 10 epochs
checkpoint_freq = None  # Set to None to disable intermediate checkpoints
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def main(args, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    use_bfloat16 = args['meta']['use_bfloat16']
    model_name = args['meta']['model_name']
    load_model = args['meta']['load_checkpoint'] or resume_preempt
    r_file = args['meta']['read_checkpoint']
    copy_data = args['meta']['copy_data']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    
    # -- Setup device (single GPU)
    if not torch.cuda.is_available():
        device = torch.device('cpu')
        logger.info("CUDA not available, using CPU")
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # -- DATA
    use_gaussian_blur = args['data']['use_gaussian_blur']
    use_horizontal_flip = args['data']['use_horizontal_flip']
    use_color_distortion = args['data']['use_color_distortion']
    color_jitter = args['data']['color_jitter_strength']
    # --
    batch_size = args['data']['batch_size']
    pin_mem = args['data']['pin_mem']
    num_workers = args['data']['num_workers']
    root_path = args['data']['root_path']
    image_folder = args['data']['image_folder']
    crop_size = args['data']['crop_size']
    crop_scale = args['data']['crop_scale']
    # -- Dataset options
    use_subset = args['data'].get('use_subset', False)
    subset_type = args['data'].get('subset_type', 'fraction')
    subset_fraction = args['data'].get('subset_fraction', 0.1)
    samples_per_class = args['data'].get('samples_per_class', 50)
    subset_file = args['data'].get('subset_file', None)
    # --

    # -- MASK
    allow_overlap = args['mask']['allow_overlap']
    patch_size = args['mask']['patch_size']
    num_enc_masks = args['mask']['num_enc_masks']
    min_keep = args['mask']['min_keep']
    enc_mask_scale = args['mask']['enc_mask_scale']
    num_pred_masks = args['mask']['num_pred_masks']
    pred_mask_scale = args['mask']['pred_mask_scale']
    aspect_ratio = args['mask']['aspect_ratio']
    # --

    # -- OPTIMIZATION
    ema = args['optimization']['ema']
    ipe_scale = args['optimization']['ipe_scale']
    wd = float(args['optimization']['weight_decay'])
    final_wd = float(args['optimization']['final_weight_decay'])
    num_epochs = args['optimization']['epochs']
    warmup = args['optimization']['warmup']
    start_lr = args['optimization']['start_lr']
    lr = args['optimization']['lr']
    final_lr = args['optimization']['final_lr']

    # -- LOGGING
    folder = args['logging']['folder']
    tag = args['logging']['write_tag']

    # Create output directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)
    
    dump = os.path.join(folder, 'params-ijepa.yaml')
    with open(dump, 'w') as f:
        yaml.dump(args, f)
    # ----------------------------------------------------------------------- #

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')
    final_path = os.path.join(folder, f'{tag}-final.pth.tar')  # Final model only
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path

    # -- make csv_logger
    csv_logger = CSVLogger(log_file,
                           ('%d', 'epoch'),
                           ('%d', 'itr'),
                           ('%.5f', 'loss'),
                           ('%.5f', 'mask-A'),
                           ('%.5f', 'mask-B'),
                           ('%d', 'time (ms)'))

    # -- init model
    encoder, predictor = init_model(
        device=device,
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name)
    target_encoder = copy.deepcopy(encoder)

    # -- make data transforms
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=pred_mask_scale,
        enc_mask_scale=enc_mask_scale,
        aspect_ratio=aspect_ratio,
        nenc=num_enc_masks,
        npred=num_pred_masks,
        allow_overlap=allow_overlap,
        min_keep=min_keep)

    transform = make_transforms(
        crop_size=crop_size,
        crop_scale=crop_scale,
        gaussian_blur=use_gaussian_blur,
        horizontal_flip=use_horizontal_flip,
        color_distortion=use_color_distortion,
        color_jitter=color_jitter)

    # -- init data-loaders (single GPU version with subset options)
    if use_subset:
        if subset_type == 'fraction':
            logger.info(f"Using fraction subset: {subset_fraction*100:.1f}% of dataset")
            _, unsupervised_loader, unsupervised_sampler = make_imagenet1k_fraction(
                transform=transform,
                batch_size=batch_size,
                fraction=subset_fraction,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=1,
                rank=0,
                root_path=root_path,
                image_folder=image_folder,
                copy_data=copy_data,
                drop_last=True,
                subset_file=subset_file)
        elif subset_type == 'balanced':
            logger.info(f"Using balanced subset: {samples_per_class} samples per class")
            _, unsupervised_loader, unsupervised_sampler = make_imagenet1k_balanced_subset(
                transform=transform,
                batch_size=batch_size,
                samples_per_class=samples_per_class,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=1,
                rank=0,
                root_path=root_path,
                image_folder=image_folder,
                copy_data=copy_data,
                drop_last=True)
        elif subset_type == 'file':
            logger.info(f"Using file subset: {subset_file}")
            _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
                transform=transform,
                batch_size=batch_size,
                collator=mask_collator,
                pin_mem=pin_mem,
                training=True,
                num_workers=num_workers,
                world_size=1,
                rank=0,
                root_path=root_path,
                image_folder=image_folder,
                copy_data=copy_data,
                drop_last=True,
                subset_file=subset_file)
        else:
            raise ValueError(f"Unknown subset_type: {subset_type}")
    else:
        logger.info("Using full ImageNet dataset")
        _, unsupervised_loader, unsupervised_sampler = make_imagenet1k(
            transform=transform,
            batch_size=batch_size,
            collator=mask_collator,
            pin_mem=pin_mem,
            training=True,
            num_workers=num_workers,
            world_size=1,
            rank=0,
            root_path=root_path,
            image_folder=image_folder,
            copy_data=copy_data,
            drop_last=True)
    ipe = len(unsupervised_loader)

    # -- init optimizer and scheduler
    logger.info(f"init optimizer and scheduler")

    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        use_bfloat16=use_bfloat16)
    
    # Freeze target encoder
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model:
        encoder, predictor, target_encoder, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    def save_checkpoint(epoch, is_final=False):
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': 1,
            'lr': lr
        }
        
        # Always save latest (for resuming)
        torch.save(save_dict, latest_path)
        
        # Save final model
        if is_final:
            torch.save(save_dict, final_path)
            logger.info(f'Saved final model to {final_path}')

    # -- TRAINING LOOP
    logger.info(f"start training : ====== ")
    logger.info(f"Total epochs: {num_epochs}")
    logger.info(f"Logging detailed stats every {epoch_log_freq} epochs")
    
    for epoch in range(start_epoch, num_epochs):
        # Simple epoch counter (always show)
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')

        # -- update data-loader epoch
        if hasattr(unsupervised_sampler, 'set_epoch'):
            unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        time_meter = AverageMeter()

        for itr, (udata, masks_enc, masks_pred) in enumerate(unsupervised_loader):

            def load_imgs():
                imgs = udata[0].to(device, non_blocking=True)
                masks_1 = [u.to(device, non_blocking=True) for u in masks_enc]
                masks_2 = [u.to(device, non_blocking=True) for u in masks_pred]
                return (imgs, masks_1, masks_2)
            
            imgs, masks_enc, masks_pred = load_imgs()
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target():
                    with torch.no_grad():
                        h = target_encoder(imgs)
                        h = F.layer_norm(h, (h.size(-1),))
                        B = len(h)
                        h = apply_masks(h, masks_pred)
                        h = repeat_interleave_batch(h, B, repeat=len(masks_enc))
                        return h

                def forward_context():
                    z = encoder(imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = F.smooth_l1_loss(z, h)
                    return loss

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                    h = forward_target()
                    z = forward_context()
                    loss = loss_fn(z, h)

                # Step 2. Backward & step
                if use_bfloat16:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                with torch.no_grad():
                    m = next(momentum_scheduler)
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (loss.item(), _new_lr, _new_wd, grad_stats)
            
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(train_step)
            loss_meter.update(loss)
            time_meter.update(etime)

            # -- Logging within epoch (only every 10 epochs)
            def log_stats():
                csv_logger.log(epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime)
                
                # Only log detailed iteration stats every epoch_log_freq epochs
                if ((epoch + 1) % epoch_log_freq == 0 or epoch == 0) and (itr % log_freq == 0):
                    if np.isnan(loss) or np.isinf(loss):
                        logger.warning(f'[{epoch + 1}, {itr:5d}] WARNING: loss is {loss}')
                    else:
                        logger.info('[%d, %5d] loss: %.3f '
                                    'masks: %.1f %.1f '
                                    '[wd: %.2e] [lr: %.2e] '
                                    '[mem: %.2e] '
                                    '(%.1f ms)'
                                    % (epoch + 1, itr,
                                       loss_meter.avg,
                                       maskA_meter.avg,
                                       maskB_meter.avg,
                                       _new_wd,
                                       _new_lr,
                                       torch.cuda.max_memory_allocated() / 1024.**2 if torch.cuda.is_available() else 0,
                                       time_meter.avg))

                        if grad_stats is not None:
                            logger.info('[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)'
                                        % (epoch + 1, itr,
                                           grad_stats.first_layer,
                                           grad_stats.last_layer,
                                           grad_stats.min,
                                           grad_stats.max))

            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- End of epoch summary
        logger.info(f'Epoch {epoch + 1}/{num_epochs} completed - avg loss: {loss_meter.avg:.4f}')
        
        # Save checkpoint
        is_final = (epoch + 1 == num_epochs)
        save_checkpoint(epoch + 1, is_final=is_final)
        
        if is_final:
            logger.info('=' * 80)
            logger.info('TRAINING COMPLETE!')
            logger.info(f'Final model saved to: {final_path}')
            logger.info(f'Final loss: {loss_meter.avg:.4f}')
            logger.info('=' * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='I-JEPA Single GPU Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    main(config, resume_preempt=args.resume)