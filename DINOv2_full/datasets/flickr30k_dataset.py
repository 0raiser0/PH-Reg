import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from utils.denoising import prepare_all_offsets, prepare_flip, prepare_patch_idxs, shuffle_shift, farthest_point_sample
import random
import numpy as np

"""To be modified"""

# Custom dataset class
class Flickr30Dataset(Dataset):
    def __init__(self, args_dict, transfrom=None):
        super().__init__()
        self.root_dir = args_dict['data_root']
        self.shift_frac = args_dict['shift_frac']
        self.input_resolution = args_dict['resolution']
        self.patch_size = args_dict['patch_size']
        self.counts = args_dict['counts']
        
        # self.num_patches = int(self.input_resolution[0] // self.patch_size)
        self.num_patches = int(self.input_resolution // self.patch_size)
        # assert self.input_resolution[0] == int(self.num_patches * self.patch_size) 
        assert self.input_resolution == int(self.num_patches * self.patch_size) 
        self.patch_idxs = prepare_patch_idxs(
            num_patches = self.num_patches
        ) # [2, num_patches, num_patches]
        
        self.transform = transfrom
        
        # self.file_list = os.listdir(self.root_dir)[:16] # debug for 1 batch
        self.file_list = os.listdir(self.root_dir)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        all_offsets_torch = prepare_all_offsets(
            num_patches=self.num_patches,
            shift_frac=self.shift_frac,
            counts=self.counts,
            seed=random.randint(1,999999999)
            # seed=42
        ) # [1, counts * 10, 2]
        
        flip = prepare_flip(
            counts=self.counts,
            seed=random.randint(1,999999999)
            # seed=42
        ) # [10, ]
        
        selected_idx = farthest_point_sample(
            all_offsets=all_offsets_torch,
            num_sample=self.counts,
            seed=random.randint(1,999999999)
        ) # [1, counts]

        selected_offsets = all_offsets_torch[0, selected_idx[0], :].numpy().astype(np.int32)
        selected_offsets[0] = [0, 0]
        # [counts, 2]
        
        # load images
        img_name = os.path.join(self.root_dir, self.file_list[index])
        img = Image.open(img_name)
        
        if self.transform is not None:
            img = self.transform(img)
        
        # Start shifting 
        shifted_imgs = []
        shifted_idxs_sets = []
        for offset_i in range(self.counts):
            # Get offset coordinate
            offset_height, offset_width = selected_offsets[offset_i]
            # Get filp, 0: no flip, 1: do flip
            do_flip = flip[offset_i]
            
            shifted_img = shuffle_shift(
                input_tensor=img,
                offset_height=offset_height*self.patch_size,
                offset_width=offset_width*self.patch_size,
                infill=-100
            )
            
            if do_flip == 1:
                # flip along horizontal dim
                shifted_img = torch.flip(shifted_img, dims=[2])
                
            # Fix out-of-bounds fill => set them to large values that match CLIP's high distribution
            # replace_values = torch.tensor([1.9303361, 2.0748837, 2.145897], device=shifted_img.device)
            replace_values = torch.tensor([2.2489083, 2.42857143, 2.64], device=shifted_img.device)
            shifted_img = torch.where(shifted_img<-50, replace_values.view(3, 1, 1), shifted_img)
            
            shifted_imgs.append(shifted_img)
            
            # Get shifted patch positions
            shifted_idxs = shuffle_shift(
                input_tensor=self.patch_idxs,
                offset_height=offset_height,
                offset_width=offset_width,
                infill=-1
            ) # [2, num_patches, num_patches]
            if do_flip == 1:
                shifted_idxs = torch.flip(shifted_idxs, dims=[2])
                
            shifted_idxs_sets.append(shifted_idxs)
            
            
        out_shifted_imgs = torch.stack(shifted_imgs, dim=0)
        out_shifted_idxs_sets = torch.stack(shifted_idxs_sets, dim=0)
            
        return img, out_shifted_imgs, out_shifted_idxs_sets
    

class RandomSquareCropPreservingShortEdge:
    def __init__(self, crop_size: int = 448):
        self.crop_size = crop_size

    def __call__(self, img: Image.Image) -> Image.Image:
        # Ensure the input image is large enough.
        w, h = img.size  # PIL size returns (width, height)
        if w < self.crop_size or h < self.crop_size:
            raise ValueError("Image size is smaller than the crop size.")

        # Decide which edge is the long edge.
        if w > h:
            # Height is the short edge (equals crop_size if pre-resized)
            # Randomly choose the left coordinate along the wide dimension.
            left = random.randint(0, w - self.crop_size)
            top = 0  # no vertical offset since height equals crop_size
        else:
            # Width is the short edge.
            left = 0  # no horizontal offset
            top = random.randint(0, h - self.crop_size)

        # Perform the crop.
        return TF.crop(img, top, left, self.crop_size, self.crop_size)   
    

def prepare_flickr30_dataloader(args_dict, mode='train'):
    assert mode in ['train', 'eval', 'test']
    
    if mode == 'train':
        img_transform = transforms.Compose([
                            transforms.Resize(args_dict['resolution'], interpolation=transforms.InterpolationMode.BICUBIC),
                            RandomSquareCropPreservingShortEdge(args_dict['resolution']), # use center crop?
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                            ])
        shuffle = True
    else:
        img_transform = transforms.Compose([
                            transforms.Resize(args_dict['resolution'], interpolation=transforms.InterpolationMode.BICUBIC),
                            transforms.CenterCrop(args_dict['resolution']),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
                            ])
        shuffle = False
        
    custom_dataset = Flickr30Dataset(
        args_dict = args_dict,
        transfrom=img_transform,
        )
    
    return custom_dataset, shuffle

