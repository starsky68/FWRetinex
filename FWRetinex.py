import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
        
class Conv(nn.Module):
    def __init__(self, in_num,out_num):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_num, out_num, 3, 1, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv(x))


        
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-7 if x.dtype == torch.float32 else 1e-5

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, conv_type='standard_conv', forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        if conv_type == 'standard_conv':
            self.partial_conv = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        elif conv_type == 'weight_standardized_conv':    
            self.partial_conv = WeightStandardizedConv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        else:
            raise NotImplementedError
            
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x):
        # for training/inference
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)

        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 dim,
                 n_div,
                 mlp_ratio,
                 drop_path,
                 layer_scale_init_value,
                 act_layer,
                 conv_type,
                 pconv_fw_type
                 ):

        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer: List[nn.Module] = [
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(dim,n_div,conv_type,pconv_fw_type)
        
        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class BasicStage(nn.Module):
    '''
    https://github.com/JierunChen/FasterNet
   '''
    def __init__(self,
                 dim,
                 depth=3,
                 n_div=2,
                 mlp_ratio=0.5,
                 drop_path=0.,
                 layer_scale_init_value=0,
                 act_layer=nn.ReLU,
                 conv_type='standard_conv',
                 pconv_fw_type='split_cat'
                 ):

        super().__init__()

        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path,
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer,
                conv_type=conv_type,
                pconv_fw_type=pconv_fw_type
            )
            for i in range(depth)
        ]

        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.blocks(x)
        return x


class FW_net(nn.Module):#fasternet weight_standardized_conv
    def __init__(self, num=64):
        super(FW_net, self).__init__()
        self.net = nn.Sequential(
            Conv(3, num),
            BasicStage(num),
            nn.Conv2d(num, 3, 3, 1, 1),
            nn.Sigmoid()
            )
            
    def forward(self, x):
        return self.net(x)


class Head(nn.Module):
    def __init__(self, num):
        super(Head, self).__init__()
        self.r = nn.Sequential(
            Conv(3, num),
            BasicStage(num),
            nn.Conv2d(num, 3, 3, 1, 1),
            nn.Sigmoid()
            )
        self.l = nn.Sequential(
            Conv(3, num),
            BasicStage(num,conv_type='weight_standardized_conv'),
            nn.Conv2d(num, 1, 3, 1, 1),
            nn.Sigmoid() 
            )
    def forward(self, x):
        return self.r(x), self.l(x)
        
class FWetinex(nn.Module):  #
    def __init__(self):
        super(FWetinex,self).__init__()        
        self.net = FW_net(num=64)  # default:64
        self.head = Head(num=64)        

    def forward(self, x):
        x = self.net(x)
        R,L = self.head(x)
        return L, R, x
