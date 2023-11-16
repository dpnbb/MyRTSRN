import torch
import torchvision
import logging
import time
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rtsrn import rtsrn
from DIV2KValidDataset import DIV2KValidDataset
from DIV2KTrainDataset import DIV2KTrainDataset

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='logs/train{}.log'.format(time.time()), level=logging.DEBUG, format=LOG_FORMAT)
# logging.debug("This is a debug log.")
# logging.info("This is a info log.")
# logging.warning("This is a warning log.")
# logging.error("This is a error log.")
# logging.critical("This is a critical log.")

writer = SummaryWriter("./logs_train")
device = torch.device("cuda")

trainDataset = DIV2KTrainDataset(root_dir='/home/reid/yupeng/dataset/DIV2K', transform=torchvision.transforms.ToTensor())
validDataset = DIV2KValidDataset(root_dir='/home/reid/yupeng/dataset/DIV2K', transform=torchvision.transforms.ToTensor())

# print(trainDataset[0])
trainDataloader = DataLoader(dataset=trainDataset, batch_size=1, shuffle=True)
validDataloader = DataLoader(dataset=validDataset, batch_size=1, shuffle=True)
model = rtsrn(4)
model = model.to(device)

print("训练数据集的长度为：{}".format(len(trainDataset)))
logging.debug("训练数据集的长度为：{}".format(len(trainDataset)))

# 损失函数
# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()
loss_fn = loss_fn.to(device)

# 优化器
# learning_rate = 0.01
# 1e-2=1 x (10)^(-2) = 1 /100 = 0.01
learning_rate = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录验证的次数
total_valid_step = 0
# 训练的轮数
epoch = 1000

resume_epoch = 0

# 恢复训练
if True:
    resume_model = "model_zoo/rtsrn_9.pth"
    model = torch.load(resume_model)
    resume_epoch = resume_model.split("_")[-1].split(".")[0]
    resume_epoch = int(resume_epoch) + 1
    total_valid_step = resume_epoch
    total_train_step = resume_epoch * len(trainDataloader)

# torch.save(model, "model_zoo/rtsrn_{}.pth".format(0))

for i in range(resume_epoch, epoch):
    print("-------第 {} 轮训练开始-------".format(i + 1))
    logging.debug("-------第 {} 轮训练开始-------".format(i + 1))

    # 训练步骤开始
    model.train()
    for data in trainDataloader:
        imgs, targets, _ = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            logging.debug("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 验证步骤开始
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for data in validDataloader:
            imgs, targets, _ = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            valid_loss = valid_loss + loss.item()

    print("验证集总Loss: {}".format(valid_loss))
    logging.debug("验证集总Loss: {}".format(valid_loss))
    writer.add_scalar("valid_loss", valid_loss, total_valid_step)
    total_valid_step = total_valid_step + 1
    torch.save(model, "model_zoo/rtsrn_{}.pth".format(i))
    print("模型{}已保存".format(i))
    logging.debug("模型{}已保存".format(i))
