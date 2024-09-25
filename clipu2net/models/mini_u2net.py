import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x.type(torch.float32))
        x = x.permute(0, 3, 1, 2)
        return x.type(orig_type)

class U2netBasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size = 3,
        stride = 1, 
        dilation: int = 1,
        bias: bool = True, 
        ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride=stride, padding=1 * dilation, dilation=dilation, bias=bias)
        self.norm = LayerNorm2d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x
    
class RSU5(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3, 
        mid_channels: int = 12, 
        out_channels: int = 3,
        bias: bool = False,
        ) -> None:
        super(RSU5, self).__init__()
        self.conv_in = U2netBasicBlock(in_channels, out_channels, bias=bias)
        
        self.conv1 = U2netBasicBlock(out_channels, mid_channels, bias=bias)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = U2netBasicBlock(mid_channels, mid_channels, bias=bias)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = U2netBasicBlock(mid_channels, mid_channels, bias=bias)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv4 = U2netBasicBlock(mid_channels, mid_channels, bias=bias)
        self.conv5 = U2netBasicBlock(mid_channels, mid_channels, bias=bias, dilation=2)

        self.conv4d = U2netBasicBlock(2 * mid_channels, mid_channels, bias=bias)
        self.conv3d = U2netBasicBlock(2 * mid_channels, mid_channels, bias=bias)
        self.conv2d = U2netBasicBlock(2 * mid_channels, mid_channels, bias=bias)
        self.conv1d = U2netBasicBlock(2 * mid_channels, out_channels, bias=bias)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        hx = x
        hxin = self.conv_in(hx)

        hx1 = self.conv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.conv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.conv4(hx)

        hx5 = self.conv5(hx4)

        hx4d = self.conv4d(torch.cat([hx5, hx4], dim=1))
        hx4dup = F.interpolate(hx4d, hx3.shape[2:], mode="bilinear", align_corners=True)

        hx3d = self.conv3d(torch.cat([hx4dup, hx3], dim=1))
        hx3dup = F.interpolate(hx3d, hx2.shape[2:], mode="bilinear", align_corners=True)

        hx2d = self.conv2d(torch.cat([hx3dup, hx2], dim=1))
        hx2dup = F.interpolate(hx2d, hx1.shape[2:], mode="bilinear", align_corners=True)

        hx1d = self.conv1d(torch.cat([hx2dup, hx1], dim=1))

        return hx1d + hxin

class RSU4(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3, 
        mid_channels: int = 12, 
        out_channels: int = 3,
        bias: bool = False,
        ) -> None:
        super(RSU4, self).__init__()
        self.conv_in = U2netBasicBlock(in_channels, out_channels, bias=bias)

        self.conv1 = U2netBasicBlock(out_channels, mid_channels, bias=bias)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv2 = U2netBasicBlock(mid_channels, mid_channels, bias=bias)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.conv3 = U2netBasicBlock(mid_channels, mid_channels, bias=bias)
        self.conv4 = U2netBasicBlock(mid_channels, mid_channels, bias=bias, dilation=2)

        self.conv3d = U2netBasicBlock(2 * mid_channels, mid_channels, bias=bias)
        self.conv2d = U2netBasicBlock(2 * mid_channels, mid_channels, bias=bias)
        self.conv1d = U2netBasicBlock(2 * mid_channels, out_channels, bias=bias)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        hx = x
        hxin = self.conv_in(hx)

        hx1 = self.conv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.conv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.conv3(hx)

        hx4 = self.conv4(hx3)

        hx3d = self.conv3d(torch.cat([hx4, hx3], dim=1))
        hx3dup = F.interpolate(hx3d, hx2.shape[2:], mode="bilinear", align_corners=True)

        hx2d = self.conv2d(torch.cat([hx3dup, hx2], dim=1))
        hx2dup = F.interpolate(hx2d, hx1.shape[2:], mode="bilinear", align_corners=True)

        hx1d = self.conv1d(torch.cat([hx2dup, hx1], dim=1))

        return hx1d + hxin

class RSU4F(nn.Module):
    def __init__(
        self, 
        in_channels=3, 
        mid_channels=12, 
        out_channels=3,
        bias: bool = False,
        ) -> None:
        super(RSU4F, self).__init__()
        self.conv_in = U2netBasicBlock(in_channels, out_channels, bias=bias)

        self.conv1 = U2netBasicBlock(out_channels, mid_channels, bias=bias)
        self.conv2 = U2netBasicBlock(mid_channels, mid_channels, bias=bias, dilation=2)
        self.conv3 = U2netBasicBlock(mid_channels, mid_channels, bias=bias, dilation=4)
        self.conv4 = U2netBasicBlock(mid_channels, mid_channels, bias=bias, dilation=8)

        self.conv3d = U2netBasicBlock(2 * mid_channels, mid_channels, bias=bias, dilation=4)
        self.conv2d = U2netBasicBlock(2 * mid_channels, mid_channels, bias=bias, dilation=2)
        self.conv1d = U2netBasicBlock(2 * mid_channels, out_channels, bias=bias)

    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        hx = x
        hxin = self.conv_in(hx)

        hx1 = self.conv1(hxin)
        hx2 = self.conv2(hx1)
        hx3 = self.conv3(hx2)

        hx4 = self.conv4(hx3)

        hx3d = self.conv3d(torch.cat([hx4, hx3], dim=1))
        hx2d = self.conv2d(torch.cat([hx3d, hx2], dim=1))
        hx1d = self.conv1d(torch.cat([hx2d, hx1], dim=1))

        return hx1d + hxin

class PrUp(nn.Module):
    """Projection upsampling layer.
    """
    def __init__(
        self, 
        d_in: int,
        d_out: int, 
        ):
        super().__init__()
        # self.proj_l = nn.Sequential(
        #     nn.Conv2d(d_in, d_out // 2, 1, bias=False),
        #     nn.GELU(),
        # )
        # self.proj_h = nn.Sequential(
        #     nn.Conv2d(d_in, d_out // 2, 1, bias=False),
        #     nn.GELU(),
        # )
        self.conv = U2netBasicBlock(d_in * 2, d_out)

    def forward(self, xl: torch.Tensor, xh: torch.Tensor):
        h, w = xh.shape[-2:]
        xl = F.interpolate(xl, size=(h, w), mode='bilinear', align_corners=True)

        # xl = self.proj_l(xl)
        # xh = self.proj_h(xh)
        x = torch.cat([xl, xh], dim=1)

        return self.conv(x)


class MiniU2Net(nn.Module):
    """CLIPU^2NETR
    Serve as a standalone module for fine-grain segmentation outputs.
    """
    def __init__(
        self,
        n_channels: int = 64,
        d_model: int = None,
        vision_patch_size: int = 16,
        ):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model

        self.proj_d = None
        if n_channels != d_model:
            self.proj_d = nn.Conv2d(d_model, n_channels, 1, bias=False)

        # Coarse: 1x1 ConvTranspose head
        self.side4 = nn.ConvTranspose2d(n_channels, 1, vision_patch_size, stride=vision_patch_size)
        # print(vision_patch_size)
        # print(self.side4.weight.data.shape)

        self.conv_in = U2netBasicBlock(3, n_channels, 3, stride=2)
        self.enc1 = RSU5(n_channels, n_channels // 4, n_channels, bias=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.prup1 = PrUp(n_channels, n_channels)

        self.enc2 = RSU4(n_channels, n_channels // 4, n_channels, bias=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.prup2 = PrUp(n_channels, n_channels)

        self.enc3 = RSU4F(n_channels, n_channels // 4, n_channels, bias=True)
        self.prup3 = PrUp(n_channels, n_channels)

        # Decoders
        self.dec3 = RSU4F(2 * n_channels, n_channels // 4, n_channels, bias=True)
        self.side3 = nn.ConvTranspose2d(n_channels, 1, vision_patch_size // 2, stride=vision_patch_size // 2)

        self.dec2 = RSU4(2 * n_channels, n_channels // 4, n_channels, bias=True)
        self.side2 = nn.ConvTranspose2d(n_channels, 1, vision_patch_size // 4, stride=vision_patch_size // 4)

        self.dec1 = RSU5(2 * n_channels, n_channels // 4, n_channels, bias=True)
        self.side1 = nn.ConvTranspose2d(n_channels, 1, vision_patch_size // 8, stride=vision_patch_size // 8)

        self.outconv = nn.Conv2d(4, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        skip: Tuple[torch.Tensor],
        ):
        height, width = x.shape[-2:]

        # Coarse head
        s4 = self.side4(skip[-1])
        # print("s4:", s4.shape)
        
        # U^2Net Encoding
        # -----
        [d1, d2, d3, d4] = skip
        hx = self.conv_in(x)
        e1 = self.enc1(hx)
        # print(e1.shape)
        e1 = self.prup1(d1, e1)
        # print(e1.shape, d1.shape)
        # print()
        # print("x:", x.shape)
        # print("enc1:", e1.shape)
        e2 = self.enc2(self.pool1(e1))
        e2 = self.prup2(d2, e2)

        # print("enc2:", e2.shape)
        e3 = self.enc3(self.pool2(e2))
        e3 = self.prup3(d3, e3)
        # print("enc3:", e3.shape)
        # print()

        # U^2Net decoding
        d4up = F.interpolate(d4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d4up, e3], dim=1))
        # print("d3:", d3.shape)
        s3 = self.side3(d3)
        # print("s3:", s3.shape)
        
        d3up = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d3up, e2], dim=1))
        s2 = self.side2(d2)
        # print("s2:", s2.shape)

        d2up = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2up, e1], dim=1))
        s1 = self.side1(d1)
        # print("s1:", s1.shape)

        # Coarse -> Fine
        if s4.shape[-1] != width:
            s4 = F.interpolate(s4, size=(height, width), mode="bilinear", align_corners=False)
            s3 = F.interpolate(s3, size=(height, width), mode="bilinear", align_corners=False)
            s2 = F.interpolate(s2, size=(height, width), mode="bilinear", align_corners=False)
            s1 = F.interpolate(s1, size=(height, width), mode="bilinear", align_corners=False)

        s0 = self.outconv(torch.cat([s4, s3, s2, s1], dim=1))
        # print("s0:", s0.shape)

        return [s4, s3, s2, s1, s0]

if __name__ == "__main__":
    model = MiniU2Net(d_model=64)
    if torch.cuda.is_available():
        model = model.cuda()
    bs = 4
    img_tensor = torch.rand(size=(bs, 3, 336, 336), dtype=torch.float32).cuda()
    f_i = torch.rand((bs, 64, 21, 21), dtype=torch.float32).cuda()
    outputs = model(img_tensor, [f_i, f_i.clone(), f_i.clone(), f_i.clone()])
    # print(outputs[-1].shape)