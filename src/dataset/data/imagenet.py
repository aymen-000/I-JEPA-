import os
import subprocess
import time
import random

import numpy as np

from logging import getLogger

import torch
import torchvision
from torch.utils.data import Subset

_GLOBAL_SEED = 0
logger = getLogger()


def create_imagenet_subset_file(imagenet_root, subset_file_path, images_per_class=50, train=True):
    """
    Create a subset file containing a limited number of images per class
    
    Args:
        imagenet_root: Path to ImageNet root directory
        subset_file_path: Path where to save the subset file
        images_per_class: Number of images to include per class
        train: Whether to create subset for training or validation data
    """
    suffix = '/' if train else 'val/'
    data_path = os.path.join(imagenet_root, suffix)
    
    subset_images = []
    
    # Iterate through each class directory
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            # Get all images in this class
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Randomly sample images_per_class images
            selected_images = random.sample(images, min(images_per_class, len(images)))
            
            # Add to subset list with format: classname_imagename.jpg
            for img in selected_images:
                subset_images.append(f"{class_name}_{img}")
    
    # Write subset file
    with open(subset_file_path, 'w') as f:
        for img in subset_images:
            f.write(f"{img}\n")
    
    print(f"Created subset file with {len(subset_images)} images")
    print(f"Saved to: {subset_file_path}")


def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False)
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file)
    logger.info('ImageNet dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ImageNet unsupervised data loader created')

    return dataset, data_loader, dist_sampler


def make_imagenet1k_fraction(
    transform,
    batch_size,
    fraction=0.1,  # Use 10% of the dataset
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    """
    Modified version that uses only a fraction of ImageNet
    """
    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False)
    
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file)
    
    # Create a random subset using only a fraction of the data
    if fraction < 1.0:
        num_samples = int(len(dataset) * fraction)
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = Subset(dataset, indices)
        logger.info(f'Using {num_samples} samples ({fraction*100:.1f}% of dataset)')
    
    logger.info('ImageNet dataset created')
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    
    logger.info('ImageNet unsupervised data loader created')
    return dataset, data_loader, dist_sampler


def make_imagenet1k_balanced_subset(
    transform,
    batch_size,
    samples_per_class=50,  # Use 50 samples per class
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True
):
    """
    Create a balanced subset with equal samples per class
    """
    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=True)  # Enable indexing for balanced sampling
    
    # Create balanced subset indices
    subset_indices = []
    for class_idx, indices in enumerate(dataset.target_indices):
        # Randomly sample samples_per_class from each class
        class_samples = random.sample(indices, min(samples_per_class, len(indices)))
        subset_indices.extend(class_samples)
    
    # Create subset
    dataset = Subset(dataset, subset_indices)
    logger.info(f'Created balanced subset with {len(subset_indices)} samples')
    logger.info(f'Approximately {samples_per_class} samples per class')
    
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    
    return dataset, data_loader, dist_sampler


class ImageNet(torchvision.datasets.ImageFolder):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_file='imagenet_full_size-061417.tar.gz',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=False,
        index_targets=False
    ):
        """
        ImageNet

        Dataset wrapper (can copy data locally to machine)

        :param root: root network directory for ImageNet data
        :param image_folder: path to images inside root network directory
        :param tar_file: zipped image_folder inside root network directory
        :param train: whether to load train data (or validation)
        :param job_id: scheduler job-id used to create dir on local machine
        :param copy_data: whether to copy data from network file locally
        :param index_targets: whether to index the id of each labeled image
        """

        suffix = '/' if train else 'val/'
        data_path = None
        if copy_data:
            logger.info('copying data locally')
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank)
        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder)
        logger.info(f'data-path {data_path}')
        
        super(ImageNet, self).__init__(root=data_path, transform=transform)
        logger.info('Initialized ImageNet')

        if index_targets:
            self.targets = []
            for sample in self.samples:
                self.targets.append(sample[1])
            self.targets = np.array(self.targets)
            self.samples = np.array(self.samples)

            mint = None
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.squeeze(np.argwhere(
                    self.targets == t)).tolist()
                self.target_indices.append(indices)
                mint = len(indices) if mint is None else min(mint, len(indices))
                logger.debug(f'num-labeled target {t} {len(indices)}')
            logger.info(f'min. labeled indices {mint}')


class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        """
        ImageNetSubset

        :param dataset: ImageNet dataset object
        :param subset_file: '.txt' file containing IDs of IN1K images to keep
        """
        self.dataset = dataset
        self.subset_file = subset_file
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        """ Filter self.dataset to a subset """
        root = self.dataset.root
        class_to_idx = self.dataset.class_to_idx
        # -- update samples to subset of IN1k targets/samples
        new_samples = []
        logger.info(f'Using {subset_file}')
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                class_name = line.split('_')[0]
                target = class_to_idx[class_name]
                img = line.split('\n')[0]
                new_samples.append(
                    (os.path.join(root, class_name, img), target)
                )
        self.samples = new_samples

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_file='imagenet_full_size-061417.tar.gz',
    job_id=None,
    local_rank=None
):
    if job_id is None:
        try:
            job_id = os.environ['SLURM_JOBID']
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    if local_rank is None:
        try:
            local_rank = int(os.environ['SLURM_LOCALID'])
        except Exception:
            logger.info('No job-id, will load directly from network file')
            return None

    source_file = os.path.join(root, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    target_file = os.path.join(target, tar_file)
    data_path = os.path.join(target, image_folder, suffix)
    logger.info(f'{source_file}\n{target}\n{target_file}\n{data_path}')

    tmp_sgnl_file = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            commands = [
                ['tar', '-xf', source_file, '-C', target]]
            for cmnd in commands:
                start_time = time.time()
                logger.info(f'Executing {cmnd}')
                subprocess.run(cmnd)
                logger.info(f'Cmnd took {(time.time()-start_time)/60.} min.')
            with open(tmp_sgnl_file, '+w') as f:
                print('Done copying locally.', file=f)
        else:
            while not os.path.exists(tmp_sgnl_file):
                time.sleep(60)
                logger.info(f'{local_rank}: Checking {tmp_sgnl_file}')

    return data_path