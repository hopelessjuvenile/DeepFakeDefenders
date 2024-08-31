import timm
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

resnet18 = models.resnet18(pretrained=True)
resnet18_feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
torch.save(resnet18_feature_extractor.state_dict(), "pretrain_weight/resnet18_no_linear.pth")
res18_weight = torch.load("resnet18_no_linear.pth")

# # 查看模型文件大小
# import os
# file_size = os.path.getsize('resnet18_no_linear.pth') / (1024 * 1024)  # 转换为MB
# print(f'Model file size: {file_size:.2f} MB')