from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import os
import random


class CycleGANDataset(Dataset):
    def __init__(self, a_dir, z_dir, file_extension='*.jpg', transform=None, aligned=False):
        super(CycleGANDataset, self).__init__()
        self.a_arr = glob.glob(os.path.join(a_dir, file_extension))
        self.z_arr = glob.glob(os.path.join(z_dir, file_extension))
        self.a_dir = a_dir
        self.z_dir = z_dir
        self.transform = transform
        self.aligned = aligned

    def __len__(self):
        # could be a arr or z arr, shouldn't matter
        return max(len(self.a_arr), len(self.z_arr))

    def __getitem__(self, index):
        """Generate one sample of data, based on dirty data"""

        # grab a random index for each. the modulus allows overflow, and makes it unaligned if overflow
        idx_a = index % len(self.a_arr)
        if self.aligned:
            idx_z = index % len(self.z_arr)
        else:
            idx_z = random.randint(0, len(self.z_arr) - 1)

        a_img_name = os.path.basename(self.a_arr[idx_a])
        z_img_name = os.path.basename(self.z_arr[idx_z])
        a_path = os.path.join(self.a_dir, a_img_name)
        z_path = os.path.join(self.z_dir, z_img_name)
        transformed_a = Image.open(a_path)
        transformed_z = Image.open(z_path)

        if self.transform:
            # notice how with each transform, they are each independent.
            # this allows the random crop and flips to be different with each img
            # because cyclegan is meant for unpaired it won't matter
            transformed_a = self.transform(transformed_a)
            transformed_z = self.transform(transformed_z)

        return {'a': transformed_a, 'z': transformed_z}


def make_training_dataloader(a_dir, z_dir, batch_size, img_h=256, img_w=256, num_workers=1, file_extension='*.jpg'):
    """
    Method to make a
    :param a_dir: domain A data directory
    :param z_dir: domain Z data directory
    :param batch_size: batch size for each batch of data from each dataset
    :param img_h: image height
    :param img_w: image width
    :param num_workers: num of cpu threads used to load data
    :param file_extension: file extension of images
    :return: a Pytorch dataloader that pulls data gradually (only data used is loaded into memory)
    """
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': num_workers}

    trans = transforms.Compose([
        transforms.Resize((int(img_h * 1.12), int(img_w * 1.12)), Image.BICUBIC),
        # make it bigger so random crop is more random
        transforms.RandomCrop((img_h, img_w)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # mean, std of each channel. can provice a tuple for 3 dim images
    ])

    return DataLoader(CycleGANDataset(a_dir, z_dir, file_extension, transform=trans), **params)
