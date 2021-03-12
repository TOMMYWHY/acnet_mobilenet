from torch import nn
import torch
import numpy as np


### V2
## 根据acnet 论文。节点自我推理(1*1)；与相邻接点推理(3*3)；全局推理(fz).根据kernel的数值实现不同的连接方式。
###12586854 # todo :计算量大4倍 org：3111462

cuda1=torch.device("cuda:1")


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def globalAvgPooling(x):
    axis = [2, 3]
    return torch.mean(x, axis)


def batch_flatten(x):
    """
    Flatten the tensor except the first dimension.
    """
    shape = x.shape[1:]
    if None not in shape:
        return torch.reshape(x, [-1, int(np.prod(shape))])
    return torch.reshape(x, torch.stack([torch.shape(x)[0], -1]))


class Fc(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Fc, self).__init__()
        # self.input = input # tensor
        self.in_channel = in_channel  # tensor
        self.out_channel = out_channel  # [1, 32, 224, 224]
        self.fc1 = nn.Linear(in_features=in_channel, out_features=self.out_channel)
        self.acv = nn.ReLU()
        # self.fc3 = nn.Linear(in_features=self.fc1.out_features, out_features= self.out_channel)

    def forward(self, x):
        x = globalAvgPooling(x)  # gap
        # print("globalAvgPooling:",x.shape)
        flatten_x = batch_flatten(x)
        # print("flatten_x:",flatten_x.shape)
        x = self.fc1(flatten_x)
        x = self.acv(x)
        # x = self.fc3(x)
        x = x.view([-1, self.out_channel, 1, 1])
        return x


# in_channel => x.shape[1] out_channel =>out_channel[2]
class Grconv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,stride,padding,groups=1):
        super(Grconv, self).__init__()
        # self.input = input
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.kernel = kernel_size
        self.padding = padding
        self.groups = groups
        self.z1 = nn.Conv2d(self.in_channel, self.out_channel, stride=self.stride, kernel_size=1, bias=False)
        self.z3 = nn.Conv2d(self.in_channel, self.out_channel, stride=self.stride, kernel_size=self.kernel, padding=self.padding,groups=self.groups,  bias=False)
        self.zf = Fc(self.in_channel, self.out_channel)
        self.soft_max = nn.Softmax(dim=0)

    def forward(self, x):
        self.p = torch.autograd.Variable(torch.ones([2, 1, x.shape[2]//self.stride, x.shape[3]//self.stride]), requires_grad=True)#.to(device=cuda1)
        self.p = self.soft_max(self.p)
        ####### local+global | z3 + zf
        z = (self.p[0:1, :, :, :] * self.z3(x) + self.p[1:2, :, :, :] * self.zf(x))
        return z




if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))
    layer = Fc(x.shape[1], [1, 32, 224, 224][1])
    z = layer(x)
    print(z.shape)
    print("Fc:", get_model_parameters(layer))  # 1184, 128
    print("-------")
    # layer_Grconv = Grconv(x.shape[1], [1, 32, 224, 224][1], kernel_size=1,stride=1,padding=0)  # input channel
    layer_Grconv = Grconv(x.shape[1], [1, 32, 224, 224][1], kernel_size=3,stride=1,padding=1)  # input channel
    # layer_Grconv = Grconv(x.shape[1], [1, 32, 224, 224][1], kernel_size=5,stride=2,padding=1)  # input channel
    z = layer_Grconv(x)
    print("Grconv:", z.shape)
    print(get_model_parameters(layer_Grconv))  # 2144  #z3x
    # todo fc -1linear 1088
    # todo Grconv -z3x

    print("1111-------")

    # cnv2 = nn.Conv2d(x.shape[1], [1, 32, 224, 224][1], 3, 1, 1, bias=False)
    cnv2 = nn.Conv2d(x.shape[1], [1, 32, 224, 224][1], stride=1, kernel_size=3, padding=1, bias=False)

    z = cnv2(x)
    print(z.shape)
    print("cnv2:", get_model_parameters(cnv2))  # 864

# Grconv: torch.Size([1, 32, 224, 224])
# 1088
# 1111-------
# torch.Size([1, 32, 224, 224])
# cnv2: 864