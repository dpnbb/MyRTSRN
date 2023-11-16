import torch
import torchvision

from rtsrn import rtsrn
from torch.utils.data import DataLoader
from DIV2KValidDataset import DIV2KValidDataset

device = torch.device("cuda")

resume_model = "model_zoo/rtsrn_9.pth"
model = rtsrn(4)
model = torch.load(resume_model)
model = model.to(device)
# print(model)
dataset = DIV2KValidDataset(root_dir='/mnt/d/dataset/DIV2K/', transform=torchvision.transforms.ToTensor())
dataLoader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
for data in dataLoader:
    lr, hr, name = data
    lr = lr.to(device)
    hr = hr.to(device)
    out = model(lr)
    torchvision.utils.save_image(out, "output/sr/{}".format(name[0]))
    torchvision.utils.save_image(hr, "output/hr/{}".format(name[0]))
    torchvision.utils.save_image(lr, "output/lr/{}".format(name[0]))
    break