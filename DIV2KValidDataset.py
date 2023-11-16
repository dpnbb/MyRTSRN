import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DIV2KValidDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        super(DIV2KValidDataset, self).__init__()
        self.root_dir = root_dir
        self.LR_dir = os.path.join(root_dir, 'DIV2K_valid_LR/')
        self.HR_dir = os.path.join(root_dir, 'DIV2K_valid_HR/')
        self.transform = transform
        self.LR_list = os.listdir(self.LR_dir)
        self.HR_list = os.listdir(self.HR_dir)
        self.LR_list.sort()
        self.HR_list.sort()

    def __getitem__(self, idx):
        lr_name = self.LR_list[idx]
        hr_name = self.HR_list[idx]
        lr_path = os.path.join(self.LR_dir, lr_name)
        hr_path = os.path.join(self.HR_dir, hr_name)
        lr_image = cv2.imread(lr_path, cv2.IMREAD_UNCHANGED)
        torch.from_numpy(np.ascontiguousarray(lr_image)).permute(2, 0, 1).float().div(255. / 255.0).unsqueeze(0)
        hr_image = cv2.imread(hr_path, cv2.IMREAD_UNCHANGED)
        torch.from_numpy(np.ascontiguousarray(hr_image)).permute(2, 0, 1).float().div(255. / 255.0).unsqueeze(0)
        if self.transform:
            lr_image = self.transform(lr_image)
            hr_image = self.transform(hr_image)
        return lr_image, hr_image, hr_name

    def __len__(self):
        assert len(self.LR_list) == len(self.HR_list)
        return len(self.LR_list)
