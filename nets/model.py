# by kayak.gw
from nets.transformer_text_encoder import text_encoder
from nets.images_encoder import img_encoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.img_encoder=img_encoder()
        self.text_encoder=text_encoder()

        self.fc1 = nn.Linear(1024, 256)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(256, 10)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)
        
    def forward(self,img,text):
        img_feature=self.img_encoder(img)
        text_feature=self.text_encoder(text)

        x=torch.add(img_feature,text_feature)
        x=F.relu(self.fc1(x))
        x=self.out(x)
        
        return x


class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.img_encoder=img_encoder()
        self.text_encoder=text_encoder()

        self.fc1 = nn.Linear(1024, 256)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(256, 10)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self,img,text):
        img_feature=self.img_encoder(img)
        text_feature=self.text_encoder(text)
        image_features = img_feature / img_feature.norm(dim=-1, keepdim=True)
        text_features = text_feature / text_feature.norm(dim=-1, keepdim=True)

        #logit_scale = self.logit_scale.exp()
        #logits_per_image = logit_scale * image_features @ text_features.t()
        #logits_per_text = logits_per_image.t()
        x = (100.0 * image_features @ text_features.T).softmax(dim=-1)*self.logit_scale
        return x

    
'''
img=torch.rand(1,3,224,224)
text=torch.rand(1,77).type(torch.LongTensor)
model=Net()
a=model(img,text)
print(a)
'''
