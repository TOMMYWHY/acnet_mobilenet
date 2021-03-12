from torch import nn
import torch
import numpy as np

def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters

def globalAvgPooling( x):
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
    def __init__(self,in_channel,out_channel):
        super(Fc, self).__init__()
        # self.input = input # tensor
        self.in_channel = in_channel # tensor
        self.out_channel = out_channel #[1, 32, 224, 224]
        self.fc1 = nn.Linear(in_features=in_channel, out_features= self.out_channel)
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

#in_channel => x.shape[1] out_channel =>out_channel[2]
class Grconv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Grconv, self).__init__()
        # self.input = input
        self.in_channel = in_channel
        self.out_channel = out_channel
        # self.filters = filters
        self.z3 = nn.Conv2d(self.in_channel, self.out_channel, stride=1, kernel_size=3, padding=1, bias=False)
        self.z1 = nn.Conv2d(self.in_channel, self.out_channel, stride=1, kernel_size=1, bias=False)
        self.zf = Fc(self.in_channel,self.out_channel)
        self.soft_max = nn.Softmax(dim=0)
        # self.nb = nn.BatchNorm2d(self.z.shape[1])
        # self.ReLU = nn.ReLU6(inplace=True)

    def forward(self, x):
        # z3x = self.z3(x)
        # z1x =self.z1(x)
        # out_channel = list(z3x.shape)
        # zfx = self.zf(x)
        self.p = torch.autograd.Variable(torch.ones([3, 1, x.shape[2], x.shape[3]]), requires_grad=True)
        # self.p = torch.autograd.Variable(torch.ones([2, 1, x.shape[2], x.shape[3]]), requires_grad=True)
        self.p =self.soft_max(self.p)
        # print("p:",self.p.shape)
        # print("p[0:1,:,:,:]  out_shape:", self.p[0:1, :, :, :])

        z = self.p[0:1, :, :, :] * self.z3(x) + self.p[1:2, :, :, :] * self.z1(x) + self.p[2:3, :, :, :] * self.zf(x)
        #1088
        # z = self.p[0:1, :, :, :] * self.zf(x)  + self.p[1:2, :, :, :] * self.z3(x) # 992
        # z = self.p[1:2, :, :, :] * self.z1(x) + self.p[1:2, :, :, :] * self.zf(x) # 224
        # z = self.p[1:2, :, :, :] * self.z1(x) + self.p[1:2, :, :, :] * self.z3(x) # 960
        # z = self.p[1:2, :, :, :] * self.zf(x) #128
        #+ self.p[1:2, :, :, :] * self.z1(x)

        return z
        # return zfx



if __name__ =="__main__":
    x = torch.rand((1, 3, 224, 224))
    layer = Fc(x.shape[1],[1, 32, 224, 224][1])
    z = layer(x)
    print(z.shape)
    print("Fc:",get_model_parameters(layer))# 1184, 128
    print("-------")
    layer_Grconv = Grconv(x.shape[1],[1, 32, 224, 224][1]) # input channel
    z = layer_Grconv(x)
    print("Grconv:",z.shape)
    print(get_model_parameters(layer_Grconv))#2144  #z3x
    # todo fc -1linear 1088
    # todo Grconv -z3x

    print("-------")

    # cnv2 = nn.Conv2d(x.shape[1], [1, 32, 224, 224][1], 3, 1, 1, bias=False)
    cnv2 = nn.Conv2d(x.shape[1], [1, 32, 224, 224][1], stride=1, kernel_size=3, padding=1, bias=False)

    z = cnv2(x)
    print(z.shape)
    print("cnv2:",get_model_parameters(cnv2))#864

#org 3111462
    #1088  fc: linear 1; Grconv: z1+z3+zf; ancnet   1*1(2) param:6505918
    #1088  fc: linear 1; Grconv: z1+z3+zf; ancnet   3*3(1) param:13164430

    #992  fc: linear 1; Grconv: z3+zf;      ancnet  3*3(1) param:12245198


    #224  fc: linear 1; Grconv: z1+zf; ancnet  1*1(2) param:3453118
    #992  fc: linear 1; Grconv: z3+zf; ancnet  1*1(2)param:6166718
    #960  fc: linear 1; Grconv: z1+z3; ancnet  1*1(2)param:6163694
