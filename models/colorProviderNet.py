import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

class WTA_scale(torch.autograd.Function):
    """
  We can implement our own custom autograd Functions by subclassing
  torch.autograd.Function and implementing the forward and backward passes
  which operate on Tensors.
  """

    @staticmethod
    def forward(ctx, input, scale=1e-4):
        """
    In the forward pass we receive a Tensor containing the input and return a
    Tensor containing the output. You can cache arbitrary Tensors for use in the
    backward pass using the save_for_backward method.
    """
        activation_max, index_max = torch.max(input, -1, keepdim=True)
        input_scale = input * scale  # default: 1e-4
        # input_scale = input * scale  # default: 1e-4
        output_max_scale = torch.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).type(torch.float)
        ctx.save_for_backward(input, mask)
        return output_max_scale

    @staticmethod
    def backward(ctx, grad_output):
        """
    In the backward pass we receive a Tensor containing the gradient of the loss
    with respect to the output, and we need to compute the gradient of the loss
    with respect to the input.
    """
        # import pdb
        # pdb.set_trace()
        input, mask = ctx.saved_tensors
        mask_ones = torch.ones_like(mask)
        mask_small_ones = torch.ones_like(mask) * 1e-4
        # mask_small_ones = torch.ones_like(mask) * 1e-4

        grad_scale = torch.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output.clone() * grad_scale
        return grad_input, None


class ResidualBlock(nn.Module):
    def __init__(self,inchannel, outchannel, kernel_size=3, padding=1, stride=1,bias=True):
        super(ResidualBlock, self).__init__()
        # 这里定义了残差块内连续的2个卷积层
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.InstanceNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=kernel_size, stride=1, padding=padding, bias=bias),
            nn.InstanceNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=bias),
                nn.InstanceNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)

        return out

class histColor_encoder(nn.Module):
    def __init__(self, block, num_block, in_channel_num=3, out_channel_num=256):
        super(histColor_encoder, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channel_num, 64, kernel_size=3, stride=2, padding=1)
        self.insNorm1 = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)               # the first layer

        self.layer1 = self._make_layer(block, 64, num_block[0], stride=1)             # four layers 2-5
        self.layer2 = self._make_layer(block, 128, num_block[1], stride=1)            # four layers 6-9
        self.layer3 = self._make_layer(block, out_channel_num, num_block[2], stride=1)            # four layers 10-13

    def _make_layer(self, block, planes, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride=stride))
            else:
                layers.append(block(planes, planes, stride=1))

        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.insNorm1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        out = self.layer3(x)
        return out


class colorProviderNet_soft(nn.Module):
    # input is Al, Bl, channel = 1, range~[0,255]
    def __init__(self, batch_size, feature_channels_num=64, inter_channels_num=256):
        super(colorProviderNet_soft, self).__init__()
        self.feature_channel = feature_channels_num
        self.in_channels = self.feature_channel * 4
        self.inter_channels = inter_channels_num
        # 44*44
        self.layer2_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=2),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )
        self.layer3_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
        )

        # 22*22->44*44
        self.layer4_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        # 11*11->44*44
        self.layer5_1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(256),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, self.feature_channel, kernel_size=3, padding=0, stride=1),
            nn.InstanceNorm2d(self.feature_channel),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )

        self.layer = nn.Sequential(
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1),
            ResidualBlock(self.feature_channel * 4, self.feature_channel * 4, kernel_size=3, padding=1, stride=1))

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

        self.upsampling = nn.Upsample(scale_factor=4)

        self.histColorEncoder = histColor_encoder(ResidualBlock, [2,2,2], out_channel_num=self.in_channels)

    def forward(self,imageX_L_norm,
                hist_tensor,
                ab_existOnImageX_tensor,
                imageX_feat=[],
                temperature=0.001 * 5,
                detach_flag=False,
                WTA_scale_weight=1,
                feature_noise=0):
        batch_size = imageX_L_norm.shape[0]
        image_height = imageX_L_norm.shape[2]
        channel_num = ab_existOnImageX_tensor.shape[1]
        image_width = imageX_L_norm.shape[3]
        feature_height = int(image_height / 4)
        feature_width = int(image_width / 4)

        # scale feature size to 64*64
        X_feature2_1 = self.layer2_1(imageX_feat[0])
        X_feature3_1 = self.layer3_1(imageX_feat[1])
        X_feature4_1 = self.layer4_1(imageX_feat[2])
        X_feature5_1 = self.layer5_1(imageX_feat[3])

        X_feature = self.layer(torch.cat((X_feature2_1, X_feature3_1, X_feature4_1, X_feature5_1), 1))
        target_feature = self.histColorEncoder(torch.cat((hist_tensor, ab_existOnImageX_tensor), 1))

        # pairwise cosine similarity
        theta = self.theta(X_feature).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        theta = theta - theta.mean(dim=-1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)
        theta_permute = theta.permute(0, 2, 1)  # 2*(feature_height*feature_width)*256
        phi = self.phi(target_feature).view(batch_size, self.inter_channels, -1)  # 2*256*(feature_height*feature_width)
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)
        # f is the correlation matrix M
        f = torch.matmul(theta_permute, phi)  # 2*(feature_height*feature_width)*(feature_height*feature_width)
        if detach_flag:
            f = f.detach()

        # f can be negative
        f_WTA = f if WTA_scale_weight == 1 else WTA_scale.apply(f, WTA_scale_weight)
        f_WTA = f_WTA / temperature
        f_div_C = F.softmax(f_WTA.squeeze_(), dim=-1)  # 2*1936*1936;

        ab_existOnImageX_tensor = F.interpolate(ab_existOnImageX_tensor, scale_factor=0.25, mode='bilinear',
                                                align_corners=True)
        colorMap_ab = ab_existOnImageX_tensor.view(batch_size, channel_num, -1)
        colorMap_ab = colorMap_ab.permute(0, 2, 1)  # 2*1936*channel

        # multiply the corr map with color
        colorProposal = torch.matmul(f_div_C, phi.permute(0, 2, 1))  # 2*1936*channel
        colorProposal = colorProposal.permute(0, 2, 1).contiguous()
        colorProposal = colorProposal.view(batch_size, self.inter_channels, feature_height, feature_width)  # 2*3*44*44
        colorProposal = self.upsampling(colorProposal)
        # similarity_map = self.upsampling(similarity_map)

        return colorProposal
