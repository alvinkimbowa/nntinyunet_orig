import os
import json
import cv2
import numpy as np
import torch
import torch.utils.data
from glob import glob

class nnUNetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, input_channels, split, fold=None, split_type='train', transform=None, eval=False):
        self.transform = transform
        self.input_channels = input_channels
        self.eval = eval
        
        nnunet_raw = os.environ['nnUNet_raw']
        nnunet_preprocessed = os.environ['nnUNet_preprocessed']
        self.img_dir = os.path.join(f'{nnunet_raw}/{dataset_name}/images{split}')
        self.label_dir = os.path.join(f'{nnunet_raw}/{dataset_name}/labels{split}')
        
        with open(os.path.join(f'{nnunet_raw}/{dataset_name}/dataset.json'), 'r') as f:
            dataset_info = json.load(f)
        self.img_ext = dataset_info['file_ending']
        
        if self.eval:
            img_ids = glob(f'{self.img_dir}/*{self.img_ext}')
            self.img_ids = [os.path.basename(img_id).replace(f'_0000{self.img_ext}', '') for img_id in img_ids]
        else:
            with open(os.path.join(f'{nnunet_preprocessed}/{dataset_name}/splits_final.json'), 'r') as f:
                splits = json.load(f)
            
            if fold == 'all':
                img_ids = []
                for split_dict in splits:
                    img_ids.extend(split_dict[split_type])
                img_ids = list(set(img_ids))
            else:
                img_ids = splits[int(fold)][split_type]
            
            self.img_ids = img_ids
        
        print("Found %d images" % len(self.img_ids))
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_filename = f'{self.img_dir}/{img_id}_0000{self.img_ext}'
        label_filename = f'{self.label_dir}/{img_id}{self.img_ext}'
        
        # Determine if other channels exist (for scenarios 2 and 4)
        other_chs = [ch for ch in glob(img_filename.replace("_0000.png", "_*.png")) if ch != img_filename]
        other_chs = sorted(other_chs)

        if self.input_channels == 3:
            if not other_chs:
                # Scenario 1: input_channels=3, only 1 image available (no other_chs)
                # Load image as RGB
                img = cv2.imread(img_filename)
            else:
                # Scenario 2: input_channels=3, all 3 images available (load all channels)
                # Load all 3 images as single-channel, concatenate
                imgs = [cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)[..., None]]
                for ch in other_chs:
                    arr = cv2.imread(ch, cv2.IMREAD_GRAYSCALE)[..., None]
                    imgs.append(arr)
                img = np.concatenate(imgs, axis=-1)
        elif self.input_channels == 1:
            if not other_chs:
                # Scenario 3: input_channels=1, only 1 image available (no other_chs)
                # Load image as RGB
                img = cv2.imread(img_filename)
            else:
                # Scenario 4: input_channels=1, all 3 images available (load all chs then convert to grayscale)
                # Load all 3 images as single-channel, concatenate and then convert to grayscale
                imgs = [cv2.imread(img_filename, cv2.IMREAD_GRAYSCALE)[..., None]]
                for ch in other_chs:
                    arr = cv2.imread(ch, cv2.IMREAD_GRAYSCALE)[..., None]
                    imgs.append(arr)
                stacked = np.concatenate(imgs, axis=-1)
                img = cv2.cvtColor(stacked, cv2.COLOR_BGR2GRAY)[..., None]
        
        if os.path.exists(label_filename):
            mask = cv2.imread(label_filename, cv2.IMREAD_GRAYSCALE)[..., None]
        elif self.eval:
            mask = np.zeros(img.shape[:2])[..., None]
        else:
            raise ValueError(f"Label file not found: {os.path.join(self.label_dir, label_filename)}")
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32')
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32')
        mask = mask.transpose(2, 0, 1)   
        
        return img, mask, {'img_id': img_id}
