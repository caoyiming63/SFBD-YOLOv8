# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from numbers import Integral

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "C1",
    "C2",
    "C2PSA",
    "C3",
    "C3TR",
    "CIB",
    "DFL",
    "ELAN1",
    "PSA",
    "SPP",
    "SPPELAN",
    "SPPF",
    "AConv",
    "ADown",
    "Attention",
    "BNContrastiveHead",
    "Bottleneck",
    "BottleneckCSP",
    "C2f",
    "C2fAttn",
    "C2fCIB",
    "C2fPSA",
    "C3Ghost",
    "C3k2",
    "C3x",
    "CBFuse",
    "CBLinear",
    "ContrastiveHead",
    "GhostBottleneck",
    "HGBlock",
    "HGStem",
    "ImagePoolingAttn",
    "Proto",
    "RepC3",
    "RepNCSPELAN4",
    "RepVGGDW",
    "ResNetLayer",
    "SCDown",
    "TorchVision",
    "Fusion",
    "DynamicShuffleUnit",
    "C2f_Faster",
    "DySample",
    "ShuffleUnit"
)

def channel_shuffle(x, groups):

    batch_size, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x

class DynamicConv2d(nn.Module):
    """
    Dynamic Convolution Layer (Dynamic Convolution 2D)

    Parameters:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int or tuple): Convolution kernel size
        stride (int or tuple): Stride, default is 1
        padding (int or tuple): Padding, default is 0
        dilation (int or tuple): Dilation factor, default is 1
        groups (int): Number of groups; default is 1
        num_kernels (int): Number of convolutional kernels used in dynamic convolution; default is 4
        reduction (int): Channel reduction ratio in the attention branch; default is 4
        bias (bool): Whether to use a bias; default is True
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, num_kernels=4, reduction=4, bias=True):
        super(DynamicConv2d, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_kernels = num_kernels
        self.reduction = reduction
        

        self.weight = nn.Parameter(
            torch.Tensor(num_kernels, out_channels, in_channels // groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_kernels, out_channels))
        else:
            self.register_parameter('bias', None)
        

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, max(in_channels // reduction, 1))
        self.fc2 = nn.Linear(max(in_channels // reduction, 1), num_kernels)
        self.softmax = nn.Softmax(dim=-1)
        

        self.reset_parameters()
    
    def reset_parameters(self):
        for i in range(self.num_kernels):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias[i], -bound, bound)
        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=math.sqrt(5))
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        """
        input:
            x (Tensor): shape (batch_size, in_channels, H, W)
        output:
            out (Tensor): shape (batch_size, out_channels, H', W')
        """
        batch_size = x.size(0)
        

        feat = self.gap(x).view(batch_size, self.in_channels)

        attn = self.fc2(F.relu(self.fc1(feat)))  # (batch, num_kernels)
        attn = self.softmax(attn) 

        
        outputs = []
        for i in range(batch_size):
            # weight_i: (out_channels, in_channels/groups, Hk, Wk)
            weight_i = torch.sum(attn[i].view(-1, 1, 1, 1, 1) * self.weight, dim=0)
            bias_i = None
            if self.bias is not None:
                bias_i = torch.sum(attn[i] * self.bias, dim=0)

            out_i = F.conv2d(
                x[i:i+1], weight_i, bias_i, 
                stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups
            )
            outputs.append(out_i)
        
        return torch.cat(outputs, dim=0)

class DynamicShuffleUnit(nn.Module):
    """
    A variant of ShuffleUnit that replaces only the three 1×1 convolutions in the downsampling branch (where stride=2) with dynamic convolutions.
    When stride=1, the original standard convolutions are retained (no replacement).
    """
    def __init__(self, input_c: int, output_c: int, stride: int,
                 num_kernels: int = 4, reduction: int = 4):
        super(DynamicShuffleUnit, self).__init__()
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
        self.stride = stride

        assert output_c % 2 == 0
        branch_features = output_c // 2


        if stride == 1:
            assert input_c == branch_features << 1


        def depthwise_conv(input_c, output_c, kernel_s, stride, padding, bias=False):
            return nn.Conv2d(in_channels=input_c, out_channels=output_c,
                             kernel_size=kernel_s, stride=stride,
                             padding=padding, bias=bias, groups=input_c)


        if self.stride == 2:

            self.branch1 = nn.Sequential(
                depthwise_conv(input_c, input_c, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(input_c),
                DynamicConv2d(
                    in_channels=input_c,
                    out_channels=branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    num_kernels=num_kernels,
                    reduction=reduction,
                    bias=False
                ),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )


            self.branch2 = nn.Sequential(
                DynamicConv2d(
                    in_channels=input_c,
                    out_channels=branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    num_kernels=num_kernels,
                    reduction=reduction,
                    bias=False
                ),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                DynamicConv2d(
                    in_channels=branch_features,
                    out_channels=branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    dilation=1,
                    groups=1,
                    num_kernels=num_kernels,
                    reduction=reduction,
                    bias=False
                ),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()
            self.branch2 = nn.Sequential(
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                depthwise_conv(branch_features, branch_features, kernel_s=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)
        return out

class PConv(nn.Module):
    def __init__(self, dim, n_div=4):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x
    
class FasterNetBlock(nn.Module):
    """
    FasterNet core building block (inverted residual structure)

    """
    def __init__(self, dim, n_div=4, expansion_ratio=2, drop_path=0.0):
        """
        Args:
            dim: Number of input/output channels
            n_div: Split ratio of PConv (default: 4, i.e., convolving 1/4 of the channels)
            expansion_ratio: Expansion ratio of PWConv
            drop_path: Drop path rate (optional)
        """
        super().__init__()
        hidden_dim = int(dim * expansion_ratio)

        self.pconv = PConv(dim, n_div)

        self.pwconv1 = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.act1 = nn.ReLU(inplace=True)

        self.pwconv2 = nn.Conv2d(hidden_dim, dim, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

        self.drop_path = nn.Identity()  

    def forward(self, x):
        shortcut = x

        # PConv
        x = self.pconv(x)

        x = self.pwconv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.pwconv2(x)
        x = self.bn2(x)

        x = shortcut + self.drop_path(x)
        return x
    
class FasterBottleneck(nn.Module):
    """
    Replacing the two 3×3 convolutions in the original Bottleneck with FasterNetBlock (expansion=2) 
    reduces the parameter count to approximately 50% of the original.
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=((3,3),(3,3)), e=1.0, n_div=4, expansion_ratio=2):
        super().__init__()
        c_ = int(c2 * e) 
        self.cv1 = FasterNetBlock(c1, n_div=n_div, expansion_ratio=expansion_ratio)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class C2f_Faster(nn.Module):
    """Replace the Bottleneck inside C2f with FasterBottleneck"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, n_div=4, expansion_ratio=2.0):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        

        self.m = nn.ModuleList(
            FasterBottleneck(self.c, self.c, shortcut, g, k=((3,3),(3,3)), e=1.0, 
                             n_div=n_div, expansion_ratio=expansion_ratio)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn']
        self.fusion = fusion
        
        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
    
    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)

class DySample(nn.Module):
    """
    equ:
        X' = grid_sample(X, S)
        S  = G + O
        O  = 0.5 * sigmoid(linear1(X)) ⊙ linear2(X)
    params:
        in_channels: C
        scale: s
        groups: g
    """
    def __init__(self, in_channels: int, scale: int = 2, groups: int = 4):
        super().__init__()
        self.scale = int(scale)
        self.groups = int(groups)
        s = self.scale
        # Output channels: 2 * g * s²  (2 represents the xy coordinates)
        out_dim = 2 * groups * (s ** 2)

        # Two independent linear branches (1×1 convolution)
        self.linear1 = nn.Conv2d(in_channels, out_dim, kernel_size=1, bias=True)
        self.linear2 = nn.Conv2d(in_channels, out_dim, kernel_size=1, bias=True)

        # Initialisation
        nn.init.normal_(self.linear1.weight, std=0.001)
        nn.init.normal_(self.linear2.weight, std=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        s = self.scale
        g = self.groups

        # -------- Step1:Generate an offset using a two-branch approach: O = 0.5*sigmoid(linear1(X)) ⊙ linear2(X) --------
        feat1 = self.linear1(x)   # (B, 2gs², H, W)
        feat2 = self.linear2(x)   # (B, 2gs², H, W)

        branch1 = 0.5 * torch.sigmoid(feat1)
        O_lowres = branch1 * feat2   # (B,2gs²,H,W)

        # -------- Step2:Apply a pixel_shuffle to offset O, increasing the resolution to  sH × sW --------
        O = F.pixel_shuffle(O_lowres, upscale_factor=s)  # (B, 2g, s*H, s*W)
        sH, sW = s * H, s * W

        # -------- Step3: G (sH × sW) --------
        grid_y, grid_x = torch.meshgrid(
            torch.arange(sH, device=x.device),
            torch.arange(sW, device=x.device),
            indexing="ij"
        )
        # G: [sH, sW, 2]
        G = torch.stack([grid_x, grid_y], dim=-1).float()
        G = G.unsqueeze(0)  # (1, sH, sW, 2)

        # (B,2g,sH,sW) -> (B,g,sH,sW,2)
        B_, C_O, H_O, W_O = O.shape
        O = O.view(B_, g, 2, H_O, W_O).permute(0,1,3,4,2) # (B, g, sH, sW, 2)

        # -------- Step4:Sampling set S = G + O --------
        G = G.unsqueeze(1) # (1,1,sH,sW,2)
        S = G + O   # (B, g, sH, sW, 2)

        # grid_sample must be normalised to the range  [-1, 1]
        S[..., 0] = 2.0 * S[..., 0] / (sW - 1) - 1.0
        S[..., 1] = 2.0 * S[..., 1] / (sH - 1) - 1.0
        S = S.to(x.dtype)

        # -------- Step5:Grouped into grid_sample; each group of channels shares a single sample set S --------
        x_group = x.view(B, g, C//g, H, W) 
        out_list = []
        for gi in range(g):
            xg = x_group[:, gi, ...]    # (B, Cg, H, W)
            sg = S[:, gi, ...]          # (B, sH, sW, 2)

            sampled = F.grid_sample(
                xg, sg,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False
            )
            out_list.append(sampled)

        out = torch.cat(out_list, dim=1) # (B, C, sH, sW)
        return out