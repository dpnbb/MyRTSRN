import torch
import torchvision
import cv2

from rtsrn import rtsrn
from torch.utils.data import DataLoader
from DIV2KValidDataset import DIV2KValidDataset

device = torch.device("cuda")

resume_model = "model_zoo/rtsrn_27.pth"
model = rtsrn(4)
model = torch.load(resume_model)
model = model.to(device)
# print(model)
dataset = DIV2KValidDataset(root_dir='/home/reid/yupeng/dataset/DIV2K',
                            transform=torchvision.transforms.ToTensor())
dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
for data in dataLoader:
    lr, hr, name = data
    lr = lr.to(device)
    hr = hr.to(device)
    out = model(lr)
    lr = lr[:, (2, 1, 0), :, :]
    hr = hr[:, (2, 1, 0), :, :]
    out = out[:, (2, 1, 0), :, :]
    # print(out.shape)
    torchvision.utils.save_image(out, "output/sr/{}".format(name[0]))
    torchvision.utils.save_image(hr, "output/hr/{}".format(name[0]))
    torchvision.utils.save_image(lr, "output/lr/{}".format(name[0]))
