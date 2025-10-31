import torch.nn.functional as F
import math
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans * patch_size ** 2, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
            # nn.PixelShuffle(patch_size),
            nn.Conv2d(out_chans, out_chans, kernel_size=kernel_size,
                      padding=kernel_size // 2, padding_mode='reflect'),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_dim * patch_size ** 2, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """

    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Sequential(nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        x = self.proj(x)
        return x


class SRU(nn.Module):
    def __init__(self, in_ch) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.gelu = nn.GELU()
        self.conv_fn = nn.Conv2d(self.in_ch, self.in_ch, 1)

        self.conv = nn.Conv2d(self.in_ch//2, self.in_ch//2, 3, 1, 1)

        self.mp = nn.MaxPool2d(3, 1, 1)
        self.linear = nn.Conv2d(self.in_ch//2, self.in_ch//2, 1)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        ori = x
        x1 = self.gelu(self.conv(x1))
        x2 = self.gelu(self.linear(self.mp(x2)))

        x = torch.cat([x1, x2], dim=1)
        x = ori + self.conv_fn(x)
        return x

class SAMU(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SAMU, self).__init__()
        # self.groups = 1
        # gc = in_channels // 4
        self.processmag = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0,groups=in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0,groups=in_channels)
        )
    def forward(self, x):
        batch, c, h, w = x.size()
        x = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x)
        pha = torch.angle(x) 
        mag = self.processmag(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        output = torch.complex(real, imag)
        output = torch.fft.irfft2(output, s=(h, w), norm='backward')
        return output


class DSMB(nn.Module):
    def __init__(
            self,
            dim,
    ):
        super(DSMB, self).__init__()
        self.dim = dim
        self.separated_hq = nn.Sequential(  # conv_init_1
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )
        self.separated_lq = nn.Sequential(  # DW
            nn.Conv2d(dim*2, dim, 1),
            nn.GELU()
        )

        self.samu = SAMU(self.dim, self.dim)
        self.alpha = nn.Parameter(torch.ones((1,dim,1,1)))
        self.belt = nn.Parameter(torch.zeros((1,dim,1,1)))
        self.linear = nn.Conv2d(dim, dim, 1, 1, 0)
        self.gelu = nn.GELU()
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

        self.sru = SRU(in_ch=dim)

        self.conv_fn = nn.Sequential(  # DW
            nn.Conv2d(dim, dim, 1),
            nn.GELU()
        )
    def forward(self, x):
        _,_,h,w=x.shape
        x_l = self.separated_lq(x)
        x_h = self.separated_hq(x)

        l_var = torch.var(x_l, dim=(-2, -1), keepdim=True)
        l_m = self.dw_conv(F.adaptive_max_pool2d(x_l, (h // 2, w // 2)))
        l_m = self.samu(l_m) + l_m
        x_l = x_l * F.interpolate(self.gelu(self.linear(l_m * self.alpha + l_var * self.belt)), size=(h,w), mode='bilinear')

        x_h = self.sru(x_h)
        
        output = self.conv_fn(x_l + x_h)        
        return output

class MSCA(nn.Module):
    def __init__(self, in_channels, branch_ratio=4):
        super().__init__()
        assert in_channels % branch_ratio == 0, "in_channels must be divisible by branch_ratio"
        gc = int(in_channels // 4)
        self.dwconv_hw = nn.Conv2d(gc, gc, 5, padding=2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=9, padding=4, groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=13, padding=6, groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_5, x_9, x_13 = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_5),
             self.dwconv_w(x_9),
             self.dwconv_h(x_13)),
            dim=1,
        )

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class StripedConv_hv(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int,
                 depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, self.kernel_size), padding=(0, self.padding), groups=in_ch if depthwise else 1),
            nn.Conv2d(in_ch, in_ch, kernel_size=(self.kernel_size, 1), padding=(self.padding, 0), groups=in_ch if depthwise else 1),
        )

    def forward(self, x):
        return self.conv(x)

class StripedBranch(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv_init = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, padding=0),
            nn.GELU(),
        )
        self.conv_fn = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        
        self.conv = StripedConv_hv(in_ch, kernel_size=kernel_size, depthwise=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.conv_init(x).chunk(2, dim=1)
        x1 = self.conv(x1)
        x = self.conv_fn(x1 * x2)
        return x

class SGFN(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int = 11):
        super().__init__()
        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.conv_init = nn.Sequential(
            nn.Conv2d(in_ch, 2*in_ch, kernel_size=1, padding=0),
            nn.GELU(),
        )
        self.conv_fn = nn.Sequential(
            nn.Conv2d(2*in_ch, in_ch, kernel_size=1, padding=0),
            nn.GELU(),
        )
        self.block = StripedBranch(in_ch=in_ch, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ori = x
        x = self.norm_1(x)
        x = self.conv_init(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.block(x1)
        x2 = self.block(x2)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_fn(x) + ori
        return x

class SSFM(nn.Module):
    def __init__(
            self,
            dim,
            fourier_decoupled=DSMB,
    ):
        super(SSFM, self).__init__()
        self.dim = dim
        self.dsmb = fourier_decoupled(dim=self.dim, se_ratio=8)

        self.channel_attention_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, padding_mode='reflect'),
            nn.GELU()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv_init = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU()
        )
        self.incep_lk = MSCA(in_channels = 2 * self.dim)


    def forward(self, x):
        x = self.conv_init(x)
        x = self.incep_lk(x)

        x = self.dsmb(x)
        x = self.channel_attention_conv(x)
        x = self.channel_attention(x) * x
        return x


class DAEB(nn.Module):
    def __init__(
            self,
            dim,
            spatio_spectral_mixer=SSFM,
            ffn=SGFN,
    ):
        super(DAEB, self).__init__()
        self.dim = dim
        self.norm1 = torch.nn.BatchNorm2d(dim)
        self.mixer = spatio_spectral_mixer(dim=self.dim)
        self.ffn = ffn(in_ch=self.dim)
    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x + copy
        x = self.ffn(x)
        return x


class UHDStage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
    ) -> None:
        super(UHDStage, self).__init__()
        self.blocks = nn.ModuleList([
            DAEB(
                dim=in_channels,
                spatio_spectral_mixer=SSFM,
                ffn=SGFN,
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class Net(nn.Module):
    def __init__(self, in_chans=3, out_chans=3, patch_size=1,
                 embed_dim=[48, 96, 192, 96, 48], depth=[2, 2, 2, 2, 2],
                 embed_kernel_size=3,
                 downsample_kernel_size=None, upsample_kernel_size=None):
        super(Net, self).__init__()
        self.patch_size = patch_size
        if downsample_kernel_size is None:
            downsample_kernel_size = 4
        if upsample_kernel_size is None:
            upsample_kernel_size = 4

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        self.layer1 = UHDStage(depth=depth[0], in_channels=embed_dim[0],
                            )
        self.skip1 = nn.Conv2d(2*embed_dim[0], embed_dim[0], 1)
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1],
                                      kernel_size=downsample_kernel_size, stride=2)
        self.layer2 = UHDStage(depth=depth[1], in_channels=embed_dim[1],
                              )
        self.skip2 = nn.Conv2d(2*embed_dim[1], embed_dim[1], 1)
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2],
                                      kernel_size=downsample_kernel_size, stride=2)
        self.layer3 = UHDStage(depth=depth[2], in_channels=embed_dim[2],
                               )
        self.upsample1 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[2], out_dim=embed_dim[3])
        self.layer4 = UHDStage(depth=depth[3], in_channels=embed_dim[3],
                               )
        self.upsample2 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[3],
                                                   out_dim=embed_dim[4])
        self.layer5 = UHDStage(depth=depth[4], in_channels=embed_dim[4],
                               )
        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, out_chans=out_chans,
                                          embed_dim=embed_dim[4], kernel_size=3)
    def forward(self, x):
        copy0 = x
        x = self.patch_embed(x)
        x = self.layer1(x)
        copy1 = x

        x = self.downsample1(x)

        x = self.layer2(x)
        copy2 = x

        x = self.downsample2(x)

        x = self.layer3(x)

        x = self.upsample1(x)

        x = self.skip2(torch.cat([x, copy2], dim=1))
        x = self.layer4(x)

        x = self.upsample2(x)

        x = self.skip1(torch.cat([x, copy1], dim=1))
        x = self.layer5(x)
        x = self.patch_unembed(x)

        x = copy0 + x
        return x

@ARCH_REGISTRY.register()
class UHDRes(nn.Module):
    def __init__(self,):
        super().__init__()
        self.restoration_network = Net(
                                        embed_dim=[12, 24, 48, 24, 12],
                                        depth=[2, 3, 4, 3, 2],
                                        embed_kernel_size=3)
    def print_network(self, model):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))

    def encode_and_decode(self, input, current_iter=None):

        restoration = self.restoration_network(input)
        return restoration

    @torch.no_grad()
    def test_tile(self, input, tile_size=240, tile_pad=16):
        # return self.test(input)
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
        """
        self.scale_factor = 1
        batch, channel, height, width = input.shape
        output_height = height * self.scale_factor
        output_width = width * self.scale_factor
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        output = input.new_zeros(output_shape)
        tiles_x = math.ceil(width / tile_size)
        tiles_y = math.ceil(height / tile_size)

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * tile_size
                ofs_y = y * tile_size
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + tile_size, width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + tile_size, height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - tile_pad, 0)
                input_end_x_pad = min(input_end_x + tile_pad, width)
                input_start_y_pad = max(input_start_y - tile_pad, 0)
                input_end_y_pad = min(input_end_y + tile_pad, height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = input[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                output_tile = self.test(input_tile)

                # output tile area on total image
                output_start_x = input_start_x * self.scale_factor
                output_end_x = input_end_x * self.scale_factor
                output_start_y = input_start_y * self.scale_factor
                output_end_y = input_end_y * self.scale_factor

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.scale_factor
                output_end_x_tile = output_start_x_tile + input_tile_width * self.scale_factor
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.scale_factor
                output_end_y_tile = output_start_y_tile + input_tile_height * self.scale_factor

                # put tile into output image
                output[:, :, output_start_y:output_end_y,
                output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                               output_start_x_tile:output_end_x_tile]
        return output

    def check_image_size(self, x, window_size=16):
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % (window_size)) % (
            window_size)
        mod_pad_w = (window_size - w % (window_size)) % (
            window_size)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # print('F.pad(x, (0, mod_pad_w, 0, mod_pad_h)', x.size())
        return x

    @torch.no_grad()
    def test(self, input):
        # _, _, h_old, w_old = input.shape
        restoration = self.encode_and_decode(input)
        output = restoration
        return output

    def forward(self, input):
        restoration = self.encode_and_decode(input)
        return restoration
