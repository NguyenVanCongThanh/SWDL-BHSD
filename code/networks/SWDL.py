import torch
from torch import nn
import torch.nn.functional as F

class DepthwiseGaussianBlur2D(nn.Module):
    def __init__(self, channels=1, kernel_size=5, sigma=1.0):
        super(DepthwiseGaussianBlur2D, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.padding = kernel_size // 2
        self.channels = channels
        self.register_buffer('kernel', self._create_gaussian_kernel(kernel_size, sigma))

    def _create_gaussian_kernel(self, kernel_size, sigma):
        with torch.no_grad():
            x = torch.arange(-self.padding, self.padding + 1, dtype=torch.float32)
            y = torch.arange(-self.padding, self.padding + 1, dtype=torch.float32)
            grid_x, grid_y = torch.meshgrid(x, y)
            kernel = torch.exp(-(grid_x.pow(2) + grid_y.pow(2)) / (2 * sigma ** 2))
            kernel = kernel / kernel.sum()
            kernel = kernel.view(1, 1, kernel_size, kernel_size)
            kernel = kernel.expand(self.channels, 1, kernel_size, kernel_size)
            kernel = kernel.cuda()
            return kernel

    def forward(self, x):
        batch_size, channels, depth, height, width = x.shape
        output = torch.zeros_like(x)
        for d in range(depth):
            slice = x[:, :, d, :, :]
            slice = F.conv2d(slice, self.kernel, padding=(self.padding, self.padding), groups=self.channels)
            output[:, :, d, :, :] = slice
        return output

class LaplacianUpsample3D(nn.Module):
    def __init__(self, scale_factor=2.0, mu=1.5):
        super(LaplacianUpsample3D, self).__init__()
        self.scale_factor = scale_factor
        self.mu = mu

    def get_levels(self, height, width, depth):
        max_dim = max(height, width, depth)
        if max_dim <= 8:
            return 0
        elif max_dim <= 32:
            return 1
        elif max_dim <= 64:
            return 2
        elif max_dim <= 128:
            return 3
        elif max_dim <= 256:
            return 4
        else:
            return 5

    def get_kernel_size(self, min_dim):
        if min_dim >= 256:
            return 5
        elif min_dim == 128:
            return 4
        elif min_dim <= 64:
            return 3
        else:
            return 3

    def build_gaussian_pyramid(self, x, levels):
        pyramid = [x]
        for _ in range(levels):
            min_dim = min(x.shape[2], x.shape[3], x.shape[4])
            kernel_size = self.get_kernel_size(min_dim)
            gaussian_blur = DepthwiseGaussianBlur2D(channels=x.shape[1], kernel_size=kernel_size, sigma=1.0)
            x = gaussian_blur(x)
            x = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True)
            pyramid.append(x)
        return pyramid

    def build_laplacian_pyramid(self, gaussian_pyramid):
        pyramid = []
        for i in range(len(gaussian_pyramid) - 1):
            min_dim = min(gaussian_pyramid[i + 1].shape[3], gaussian_pyramid[i + 1].shape[4])
            kernel_size = self.get_kernel_size(min_dim)
            gaussian_blur = DepthwiseGaussianBlur2D(channels=gaussian_pyramid[i + 1].shape[1], kernel_size=kernel_size,
                                                    sigma=1.0)
            upsampled = F.interpolate(gaussian_pyramid[i + 1], size=gaussian_pyramid[i].shape[2:], mode='trilinear',
                                      align_corners=True)
            upsampled = gaussian_blur(upsampled)
            laplacian = gaussian_pyramid[i] - upsampled
            pyramid.append(laplacian)
        pyramid.append(gaussian_pyramid[-1])
        return pyramid

    def reconstruct_image(self, laplacian_pyramid, target_size, mu):
        image = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            min_dim = min(laplacian_pyramid[i].shape[2], laplacian_pyramid[i].shape[3], laplacian_pyramid[i].shape[4])
            kernel_size = self.get_kernel_size(min_dim)
            gaussian_blur = DepthwiseGaussianBlur2D(channels=laplacian_pyramid[i].shape[1], kernel_size=kernel_size,
                                                    sigma=1.0)
            upsampled = F.interpolate(image, size=laplacian_pyramid[i].shape[2:], mode='trilinear', align_corners=True)
            upsampled = gaussian_blur(upsampled)
            if upsampled.shape != laplacian_pyramid[i].shape:
                raise ValueError(f"Shape mismatch: upsampled {upsampled.shape}, laplacian {laplacian_pyramid[i].shape}")
            image = upsampled + mu * laplacian_pyramid[i]
        image = F.interpolate(image, size=target_size, mode='trilinear', align_corners=True)
        return image

    def forward(self, x):
        _, channels, height, width, depth = x.shape
        levels = self.get_levels(height, width, depth)

        if levels == 0:
            return F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=True)
        else:
            target_size = (
            int(height * self.scale_factor), int(width * self.scale_factor), int(depth * self.scale_factor))
            gaussian_pyramid = self.build_gaussian_pyramid(x, levels)
            laplacian_pyramid = self.build_laplacian_pyramid(gaussian_pyramid)
            output = self.reconstruct_image(laplacian_pyramid, target_size, self.mu)
            return output


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()
        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()
        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling_function(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none', mode_upsampling=1):
        super(Upsampling_function, self).__init__()
        ops = []
        if mode_upsampling == 0:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
        if mode_upsampling == 1:
            ops.append(LaplacianUpsample3D(scale_factor=stride, mu=1.5))
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))

        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.dropout = nn.Dropout3d(p=0.3, inplace=False)

    def forward(self, input, en=[]):
        if len(en) != 0:
            x1 = self.block_one(input)
            x1 = x1 + en[4]
            x1_dw = self.block_one_dw(x1)

            x2 = self.block_two(x1_dw)
            x2 = x2 + en[3]
            x2_dw = self.block_two_dw(x2)

            x3 = self.block_three(x2_dw)
            x3 = x3 + en[2]
            x3_dw = self.block_three_dw(x3)

            x4 = self.block_four(x3_dw)
            x4 = x4 + en[1]
            x4_dw = self.block_four_dw(x4)

            x5 = self.block_five(x4_dw)
            x5 = x5 + en[0]

            if self.has_dropout:
                x5 = self.dropout(x5)
        else:
            x1 = self.block_one(input)
            x1_dw = self.block_one_dw(x1)

            x2 = self.block_two(x1_dw)
            x2_dw = self.block_two_dw(x2)

            x3 = self.block_three(x2_dw)
            x3_dw = self.block_three_dw(x3)

            x4 = self.block_four(x3_dw)
            x4_dw = self.block_four_dw(x4)

            x5 = self.block_five(x4_dw)

            if self.has_dropout:
                x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        return res


class Decoder(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False, up_type=0):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_five_up = Upsampling_function(n_filters * 16, n_filters * 8, normalization=normalization,
                                                 mode_upsampling=up_type)
        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up = Upsampling_function(n_filters * 8, n_filters * 4, normalization=normalization,
                                                mode_upsampling=up_type)
        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up = Upsampling_function(n_filters * 4, n_filters * 2, normalization=normalization,
                                                  mode_upsampling=up_type)
        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up = Upsampling_function(n_filters * 2, n_filters, normalization=normalization,
                                                  mode_upsampling=up_type)
        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, features, f1='none', f2='none'):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        if f1 == 'none' and f2 == 'none':
            x5_up_ori = self.block_five_up(x5)
            x5_up = x5_up_ori + x4

            x6 = self.block_six(x5_up)
            x6_up_ori = self.block_six_up(x6)
            x6_up = x6_up_ori + x3

            x7 = self.block_seven(x6_up)
            x7_up_ori = self.block_seven_up(x7)
            x7_up = x7_up_ori + x2

            x8 = self.block_eight(x7_up)
            x8_up_ori = self.block_eight_up(x8)
            x8_up = x8_up_ori + x1

            x9 = self.block_nine(x8_up)
            if self.has_dropout:
                x9 = self.dropout(x9)
            out_seg = self.out_conv(x9)

        elif f1 != 'none' and f2 == 'none':
            m5, m4, m3, m2, m1 = f1[0], f1[1], f1[2], f1[3], f1[4]
            w5, w4, w3, w2, w1 = torch.sigmoid(m5), torch.sigmoid(m4), torch.sigmoid(m3), torch.sigmoid(
                m2), torch.sigmoid(m1)
            w5, w4, w3, w2, w1 = w5.detach(), w4.detach(), w3.detach(), w2.detach(), w1.detach()
            x5 = x5 + x5 * w5
            x5_up_ori = self.block_five_up(x5)
            x5_up = x5_up_ori + x4 * w4

            x6 = self.block_six(x5_up)
            x6_up_ori = self.block_six_up(x6)
            x6_up = x6_up_ori + x3 * w3

            x7 = self.block_seven(x6_up)
            x7_up_ori = self.block_seven_up(x7)
            x7_up = x7_up_ori + x2 * w2

            x8 = self.block_eight(x7_up)
            x8_up_ori = self.block_eight_up(x8)
            x8_up = x8_up_ori + x1 * w1

            x9 = self.block_nine(x8_up)
            if self.has_dropout:
                x9 = self.dropout(x9)
            out_seg = self.out_conv(x9)
        return out_seg, [x5, x5_up_ori, x6_up_ori, x7_up_ori, x8_up_ori]


class SideConv(nn.Module):
    def __init__(self, n_classes=2):
        super(SideConv, self).__init__()
        self.side5 = nn.Conv3d(256, n_classes, 1, padding=0)
        self.side4 = nn.Conv3d(128, n_classes, 1, padding=0)
        self.side3 = nn.Conv3d(64, n_classes, 1, padding=0)
        self.side2 = nn.Conv3d(32, n_classes, 1, padding=0)
        self.side1 = nn.Conv3d(16, n_classes, 1, padding=0)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, stage_feat):
        x5, x5_up, x6_up, x7_up, x8_up = stage_feat[0], stage_feat[1], stage_feat[2], stage_feat[3], stage_feat[4]
        out5 = self.side5(x5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)
        out5 = self.upsamplex2(out5)

        out4 = self.side4(x5_up)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)
        out4 = self.upsamplex2(out4)

        out3 = self.side3(x6_up)
        out3 = self.upsamplex2(out3)
        out3 = self.upsamplex2(out3)

        out2 = self.side2(x7_up)
        out2 = self.upsamplex2(out2)

        out1 = self.side1(x8_up)
        return [out5, out4, out3, out2, out1]


class SWDL_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=2, n_filters=16, normalization='none', has_dropout=False,
                 has_residual=False):
        super(SWDL_Net, self).__init__()
        self.encoder = Encoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual)
        self.decoder1 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 0)
        self.decoder2 = Decoder(n_channels, n_classes, n_filters, normalization, has_dropout, has_residual, 1)
        self.sideconv1 = SideConv()

    def forward(self, input, en):
        features = self.encoder(input, en)
        out_seg1, stage_feat1 = self.decoder1(features)
        out_seg2, stage_feat2 = self.decoder2(features, stage_feat1)
        deep_out1 = self.sideconv1(stage_feat1)
        return out_seg1, out_seg2, [stage_feat2, stage_feat1], deep_out1, []