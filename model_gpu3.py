import pywt
import torch
import torch.nn as nn
import torchvision.ops
import math
import utils.utils2
import PixelAlign2 as PixelAlign2
import torch.nn.functional as F
from einops import rearrange


class Net(nn.Module):
    def __init__(self, angRes_in, angRes_out):
        super(Net, self).__init__()
        channels = 64
        n_group = 4
        n_block = 4
        self.angRes_in = angRes_in  # 2
        self.angRes_out = angRes_out  # 7
        self.init_conv = nn.Conv2d(1, channels, kernel_size=3, stride=1, dilation=angRes_in, padding=angRes_in,
                                   bias=False)

        self.HFEGroup = HFEGroup( n_group, n_block, angRes_in, channels)
        self.UpSample = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=angRes_in, stride=angRes_in, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels * angRes_out * angRes_out, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(angRes_out),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, dilation=angRes_out, padding=angRes_out, bias=False)
        )
        self.residual = PixelAlign2.Pixel_align(1)


    def forward(self, x):  # 前向传播
        x = SAI2MacPI(x, self.angRes_in)
        buffer = self.init_conv(x)
        buffer = self.HFEGroup(buffer)

        out = self.UpSample(buffer)
        out = self.residual(out)
        out = MacPI2SAI(out, self.angRes_out)
        return out


class HFEGroup(nn.Module):
    def __init__(self, n_group, n_block, angRes, channels):  # 4,4,2,64
        super(HFEGroup, self).__init__()
        self.n_group = n_group
        Groups = []
        for i in range(n_group):
            Groups.append(HFEGroup1(n_block, angRes, channels))
        self.Group = nn.Sequential(*Groups)
        self.fuse = nn.Conv2d(n_group * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        temp = []
        for i in range(self.n_group):
            x = self.Group[i](x)
            temp.append(x)
        out = torch.cat(temp, dim=1)
        return self.fuse(out)


class HFEGroup1(nn.Module):
    def __init__(self, n_block, angRes, channels):  # 4,2,64
        super(HFEGroup1, self).__init__()
        self.n_block = n_block
        self.angRes = angRes

        self.GlobalBlocks = nn.ModuleList([GlobalBlock(angRes, channels) for _ in range(n_block)])
        self.CnnBlocks = nn.ModuleList([CnnBlock(angRes, channels) for _ in range(n_block)])
        self.FreqEnhancers = nn.ModuleList([FreqEhancer(channels) for _ in range(n_block)])

        self.fuse_global = nn.Conv2d((n_block + 1) * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.fuse_cnn = nn.Conv2d((n_block + 1) * channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(2*channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False))

    def forward(self, x):
        x0 = x
        global_feats = [x0]
        cnn_feats = [x0]

        for i in range(self.n_block):
            gi = self.GlobalBlocks[i](cnn_feats[i])
            fre = self.FreqEnhancers[i](gi)
            cx = self.CnnBlocks[i](cnn_feats[i], gi, fre)
            cnn_feats.append(cx)
            global_feats.append(gi)

        gout_in = torch.cat(global_feats, dim=1)  # [x0, g1, g2, ..., gn]
        gout = self.fuse_global(gout_in)

        # -------- Final fuse --------
        final_in = torch.cat(cnn_feats, dim=1)  # n_block 个 CNN 特征 + 1 个全局融合
        cout = self.fuse_cnn(final_in)
        out = self.fuse(torch.cat([cout, gout], dim=1))
        return out


class GlobalBlock(nn.Module):
    def __init__(self, angRes, channels):
        super(GlobalBlock, self).__init__()
        self.angRes = angRes
        self.epi_trans = BasicTrans(channels, channels * 2)

        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
        )

        self.AngConvSq = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),
                      bias=False),
        )

    def forward(self, x):
        # 2、 长距离依赖
        epi_trans = rearrange(x, 'b c (u h) (v w) -> b c (u v) h w', u=self.angRes, v=self.angRes)
        [_, _, _, h, w] = epi_trans.size()
        # Horizontal
        buffer_h = rearrange(epi_trans, 'b c (u v) h w -> b c (v w) u h', u=self.angRes, v=self.angRes)
        buffer_h = self.epi_trans(buffer_h)
        buffer_h = rearrange(buffer_h, 'b c (v w) u h -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer_h = self.conv(buffer_h) + epi_trans
        buffer_h = rearrange(buffer_h, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)
        # Vertical
        buffer_v = rearrange(epi_trans, 'b c (u v) h w -> b c (u h) v w', u=self.angRes, v=self.angRes)
        buffer_v = self.epi_trans(buffer_v)
        buffer_v = rearrange(buffer_v, 'b c (u h) v w -> b c (u v) h w', u=self.angRes, v=self.angRes, h=h, w=w)
        buffer_v = self.conv(buffer_v) + epi_trans
        buffer_v = rearrange(buffer_v, 'b c (u v) h w -> b c (u h) (v w)', u=self.angRes, v=self.angRes)

        buffer_1 = torch.cat((buffer_h, buffer_v), dim=1)
        buffer_1 = self.AngConvSq(buffer_1)
        return buffer_1



class  FreqEhancer(nn.Module):
    def __init__(self, channels):
        super(FreqEhancer, self).__init__()


        # FFT
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))

        # 可学习的 frequency_threshold
        self.frequency_threshold = nn.Parameter(torch.tensor(12.0))  # 初始值设置为10

    def forward(self, x):
        # 3、频率增强

        freq = torch.fft.fft2(x)
        magnitude = torch.abs(freq)
        h, w = x.shape[-2], x.shape[-1]
        cx, cy = w // 2, h // 2  # 中心低频
        high_freq_mask = torch.ones_like(magnitude)
        threshold = self.frequency_threshold.item()
        high_freq_mask[:, :, cy - int(threshold):cy + int(threshold),
        cx - int(threshold):cx + int(threshold)] = 0  # mask  低频切除
        high_freq = freq * high_freq_mask * int(threshold) / 10
        enhanced_freq = high_freq * self.weight  # 权重增强高频
        spa_freq = torch.abs(torch.fft.ifft2(enhanced_freq))
        return spa_freq


class CnnBlock(nn.Module):
    def __init__(self, angRes, channels, first=False):  # 2,64
        super(CnnBlock, self).__init__()
        SpaChannel, AngChannel, EpiChannel = channels, channels, channels


        self.angRes = angRes


        self.squeezeConv = nn.Sequential(
            nn.Conv2d( 2*SpaChannel + AngChannel + EpiChannel, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.SiLU(inplace=True),
        )

        self.SpaConv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, dilation=int(angRes), padding=int(angRes),bias=False),
        )

        # 替代Sobel：使用学习式边缘提取器（3x3卷积）
        self.edge_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3,  dilation=int(angRes), padding=int(angRes), bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3,  dilation=int(angRes), padding=int(angRes), bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()  # 归一化为 attention mask
        )

        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.Sigmoid()
        )


    def forward(self, x, buffer_hv, spa_freq):

        # 1、边缘特征
        edge_mask = self.edge_conv(x)  # [B, C, H, W], 0~1
        edge_enhanced = edge_mask * x
        concat = torch.cat([x, edge_enhanced], dim=1)  # [B, 2C, H, W]
        buffer_edge = self.gate(concat) * x + (1 - self.gate(concat)) * edge_enhanced

        #  2、 特征融合
        buffer = torch.cat((x, buffer_edge, buffer_hv, spa_freq), dim=1)
        buffer = self.squeezeConv(buffer)
        y = self.SpaConv(buffer) + buffer

        return y






class BasicTrans(nn.Module):
    def __init__(self, channels, spa_dim, num_heads=8):
        super(BasicTrans, self).__init__()
        self.linear_in = nn.Linear(channels, spa_dim, bias=False)
        self.norm = nn.LayerNorm(spa_dim)
        self.attention = nn.MultiheadAttention(spa_dim, num_heads, bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(spa_dim),
            nn.Linear(spa_dim, spa_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(spa_dim * 2, spa_dim, bias=False)

        )
        self.linear_out = nn.Linear(spa_dim, channels, bias=False)

    def forward(self, buffer):
        [_, _, n, v, w] = buffer.size()

        epi_token = rearrange(buffer, 'b c n v w  -> (v w) (b n) c')
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)

        return buffer


class PixelShuffle1D(nn.Module):
    """
    1D pixel shuffler
    Upscales the last dimension (i.e., W) of a tentor by reducing its channel length
    inout: x of size [b, factor*c, h, w]
    output: y of size [b, c, h, w*factor]
    """

    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor
        x = x.contiguous().view(b, self.factor, c, h, w)
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # b, c, h, w, factor
        y = x.view(b, c, h, w * self.factor)
        return y


def MacPI2SAI(x, angRes):
    out = []
    for i in range(angRes):
        out_h = []
        for j in range(angRes):
            out_h.append(x[:, :, i::angRes, j::angRes])
        out.append(torch.cat(out_h, 3))
    out = torch.cat(out, 2)
    return out


def SAI2MacPI(x, angRes):
    b, c, hu, wv = x.shape
    h, w = hu // angRes, wv // angRes
    tempU = []
    for i in range(h):
        tempV = []
        for j in range(w):
            tempV.append(x[:, :, i::h, j::w])
        tempU.append(torch.cat(tempV, dim=3))
    out = torch.cat(tempU, dim=2)
    return out


if __name__ == "__main__":
    net = Net(angRes_in=2, angRes_out=7).cuda()
    from thop import profile

    input = torch.randn(1, 1, 128, 128).cuda()
    flops, params = profile(net, inputs=(input,))


    print('   Number of parameters: %.2fM' % (params / 1024 ** 2))
    print('   Number of FLOPs: %.2fG' % (flops / 1024 ** 3))