# by kayak.gw
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils

import numpy as np
import torch.optim as optim
from nets.model import net,Net


img=torch.rand(8,3,224,224)
text=torch.rand(1,77).type(torch.LongTensor)
model=net()
a=model(img,text)
print(a)
