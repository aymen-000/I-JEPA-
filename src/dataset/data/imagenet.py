import os
import random
from logging import getLogger
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from datasets import load_dataset

_GLOBAL_SEED = 0
logger = getLogger()


def create_imagenet_subset_file(imagenet_root, subset_file_path, images_per_class=50, train=True):
    """
    Create a subset file containing a limited number of images per class
    Note: This function is kept for compatibility but not actively used with Tiny ImageNet from HuggingFace
    
    Args:
        imagenet_root: Path to ImageNet root directory
        subset_file_path: Path where to save the subset file
        images_per_class: Number of images to include per class
        train: Whether to create subset for training or validation data
    """
    logger.warning("create_imagenet_subset_file is not implemented for Tiny ImageNet from HuggingFace")
    pass


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
    """Load Tiny ImageNet from HuggingFace"""
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
    fraction=0.1,
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
    Modified version that uses only a fraction of Tiny ImageNet
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
    samples_per_class=50,
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
        index_targets=True)
    
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


class ImageNet(Dataset):
    """Tiny ImageNet dataset loaded from HuggingFace"""

    def __init__(
        self,
        root=None,
        image_folder=None,
        tar_file=None,
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=False,
        index_targets=False
    ):
        """
        Tiny ImageNet from HuggingFace
        
        :param root: Not used (kept for compatibility)
        :param image_folder: Not used (kept for compatibility)
        :param tar_file: Not used (kept for compatibility)
        :param transform: transformations to apply to images
        :param train: whether to load train data (or validation)
        :param job_id: Not used (kept for compatibility)
        :param copy_data: Not used (kept for compatibility)
        :param index_targets: whether to index the id of each labeled image
        """
        
        self.transform = transform
        self.train = train
        
        # Load Tiny ImageNet from HuggingFace
        split = 'train' if train else 'valid'
        logger.info(f'Loading Tiny ImageNet from HuggingFace (split: {split})')
        self.hf_dataset = load_dataset('Maysee/tiny-imagenet', split=split)
        logger.info(f'Loaded {len(self.hf_dataset)} images')
        
        # Get classes
        self.classes = list(set(self.hf_dataset['label']))
        self.classes.sort()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Create samples list for compatibility
        self.samples = []
        self.targets = []
        for idx in range(len(self.hf_dataset)):
            item = self.hf_dataset[idx]
            label = item['label']
            target = self.class_to_idx[label]
            self.samples.append((idx, target))  # Store index instead of path
            self.targets.append(target)
        
        logger.info('Initialized Tiny ImageNet from HuggingFace')

        if index_targets:
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

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        item = self.hf_dataset[index]
        img = item['image']
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        label = item['label']
        target = self.class_to_idx[label]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


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
        """Filter self.dataset to a subset"""
        # For Tiny ImageNet from HuggingFace, we'll use indices directly
        new_samples = []
        logger.info(f'Using {subset_file}')
        
        # Read subset indices from file
        with open(subset_file, 'r') as rfile:
            for line in rfile:
                try:
                    idx = int(line.strip())
                    if idx < len(self.dataset):
                        target = self.dataset.targets[idx]
                        new_samples.append((idx, target))
                except ValueError:
                    logger.warning(f'Invalid index in subset file: {line.strip()}')
        
        self.samples = new_samples
        logger.info(f'Filtered to {len(self.samples)} samples')

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        idx, target = self.samples[index]
        img, _ = self.dataset[idx]
        return img, target


def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_file='imagenet_full_size-061417.tar.gz',
    job_id=None,
    local_rank=None
):
    """
    Not used for Tiny ImageNet from HuggingFace
    Kept for compatibility
    """
    logger.info('copy_imgnt_locally not needed for HuggingFace datasets')
    return None


# Example usage
def example_usage():
    """Example of how to use the dataset"""
    from torchvision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset and dataloader
    dataset, data_loader, dist_sampler = make_imagenet1k(
        transform=transform,
        batch_size=32,
        num_workers=2,
        training=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    # Get a sample
    img, label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Label: {label}")


if __name__ == "__main__":
    example_usage()