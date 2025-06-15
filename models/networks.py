import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='xavier', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='xavier', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def define_G(input_nc, output_nc, norm='batch', init_type='xavier', gpu_ids=[], use_tanh=True):
    norm_layer = get_norm_layer(norm_type=norm)
    netG = UNetGenerator(input_nc, output_nc, norm_layer=norm_layer, use_tanh=use_tanh)
    return init_net(netG, init_type, gpu_ids)


class HistogramNet(nn.Module):
    def __init__(self,bin_num):
        super(HistogramNet,self).__init__()
        self.bin_num = bin_num
        self.LHConv_1 = BiasedConv1(1,bin_num)
        self.relu = nn.ReLU(True)

    def forward(self,input):
        a1 = self.LHConv_1(input)
        a2 = torch.abs(a1)
        a3 = 1- a2*(self.bin_num-1)
        a4 = self.relu(a3)
        return a4

    def getBiasedConv1(self):
        return self.LHConv_1

    def getBin(self):
        return self.bin_num

    def init_biased_conv1(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m.bias.data = -torch.arange(-1,1,2/(self.bin_num), requires_grad=False)
            m.weight.data = torch.ones(self.bin_num,1,1,1, requires_grad=False)

class BiasedConv1(nn.Module):
    def __init__(self,dim_in,dim_out):
        super(BiasedConv1, self).__init__()
        model = []
        model += [nn.Conv2d(dim_in,dim_out,kernel_size=1,padding=0,stride=1,bias=True),]
        self.model = nn.Sequential(*model)

    def forward(self,input):
        a = self.model(input)
        return a

class HistogramLoss_2channel(nn.Module):
    def __init__(self, bin_num=256, delta=0.01, imSize=256):
        super(HistogramLoss_2channel, self).__init__()
        self.imSize = imSize
        self.delta = delta
        self.bin_num = bin_num
        self.hist_net = self.define_Hist().to(device)
        self.criterionHuber = HuberLoss(delta=self.delta)

    def forward(self, inp_image, ref_image):
        hist_inp = self.getHistogram2d_conv(inp_image)
        hist_gt = self.getHistogram2d_conv(ref_image)
        return self.criterionHuber(hist_inp, hist_gt)

    def define_Hist(self):
        netHist = HistogramNet(self.bin_num)
        netHist.getBiasedConv1().apply(netHist.init_biased_conv1)
        netHist.eval()

        return netHist

    def getHistogram2d_conv(self, tensor):
        # Preprocess
        hist_a = self.hist_net((tensor[:, 0, :, :].unsqueeze(1) + 1) / 2)
        hist_b = self.hist_net((tensor[:, 1, :, :].unsqueeze(1) + 1) / 2)

        # Network
        BIN = self.hist_net.getBin()
        tensor1 = hist_a.repeat(1, BIN, 1, 1)
        tensor2 = hist_b.repeat(1, 1, BIN, 1).view(-1, BIN * BIN, self.imSize, self.imSize)

        pool = nn.AvgPool2d(self.imSize)
        hist2d = pool(tensor1 * tensor2)
        hist2d = hist2d.view(-1, 1, BIN, BIN)

        return hist2d

class Gradient_Net(nn.Module):
  def __init__(self, channel_num=3):
    super(Gradient_Net, self).__init__()
    kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).expand((channel_num,1,3,3)).to(device)

    kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).expand((channel_num,1,3,3)).to(device)

    self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
    self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

  def forward(self, x):
    grad_x = F.conv2d(x, weight=self.weight_x,groups=x.shape[1],bias=None,padding=1)
    grad_y = F.conv2d(x, weight=self.weight_y,groups=x.shape[1],bias=None,padding=1)
    gradient = torch.abs(grad_x) + torch.abs(grad_y)
    #gradient = torch.sqrt(grad_x**2+grad_y**2)
    return gradient

class GradientLoss(nn.Module):
    def __init__(self, delta=.01, channel_num=3):
        super(GradientLoss, self).__init__()
        self.channel_num = channel_num
        self.delta = delta
        self.gradient_model = Gradient_Net(channel_num=self.channel_num)
        self.criterionHuber = HuberLoss(delta=self.delta)

    def __call__(self, in0, in1):
        g1 = self.gradient_model(in0)
        g2 = self.gradient_model(in1)
        return self.criterionHuber(g1,g2)

class HuberLoss(nn.Module):
    def __init__(self, delta=.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def __call__(self, in0, in1):
        mask = torch.zeros_like(in0)
        mann = torch.abs(in0 - in1)
        eucl = .5 * (mann**2)
        mask[...] = mann < self.delta

        # loss = eucl*mask + self.delta*(mann-.5*self.delta)*(1-mask)
        loss = eucl * mask / self.delta + (mann - .5 * self.delta) * (1 - mask)
        return torch.sum(loss, dim=1, keepdim=True)

class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.sum(torch.abs(in0 - in1), dim=1, keepdim=True)

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def __call__(self, in0, in1):
        return torch.sum((in0 - in1)**2, dim=1, keepdim=True)

class UNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_tanh=True):
        super(UNetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        use_bias = True

        # Conv1
        # model1=[nn.ReflectionPad2d(1),]
        model1 = [nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # model1+=[norm_layer(64),]
        model1 += [nn.ReLU(True), ]
        # model1+=[nn.ReflectionPad2d(1),]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model1 += [nn.ReLU(True), ]
        model1 += [norm_layer(64), ]
        # add a subsampling operation

        # Conv2
        # model2=[nn.ReflectionPad2d(1),]
        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # model2+=[norm_layer(128),]
        model2 += [nn.ReLU(True), ]
        # model2+=[nn.ReflectionPad2d(1),]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model2 += [nn.ReLU(True), ]
        model2 += [norm_layer(128), ]
        # add a subsampling layer operation

        # Conv3
        # model3=[nn.ReflectionPad2d(1),]
        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # model3+=[norm_layer(256),]
        model3 += [nn.ReLU(True), ]
        # model3+=[nn.ReflectionPad2d(1),]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # model3+=[norm_layer(256),]
        model3 += [nn.ReLU(True), ]
        # model3+=[nn.ReflectionPad2d(1),]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model3 += [nn.ReLU(True), ]
        model3 += [norm_layer(256), ]
        # add a subsampling layer operation

        # Conv4
        # model47=[nn.ReflectionPad2d(1),]
        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # model4+=[norm_layer(512),]
        model4 += [nn.ReLU(True), ]
        # model4+=[nn.ReflectionPad2d(1),]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # model4+=[norm_layer(512),]
        model4 += [nn.ReLU(True), ]
        # model4+=[nn.ReflectionPad2d(1),]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model4 += [nn.ReLU(True), ]
        model4 += [norm_layer(512), ]

        # Conv5
        # model47+=[nn.ReflectionPad2d(2),]
        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        # model5+=[norm_layer(512),]
        model5 += [nn.ReLU(True), ]
        # model5+=[nn.ReflectionPad2d(2),]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        # model5+=[norm_layer(512),]
        model5 += [nn.ReLU(True), ]
        # model5+=[nn.ReflectionPad2d(2),]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model5 += [nn.ReLU(True), ]
        model5 += [norm_layer(512), ]

        # Conv6
        # model6+=[nn.ReflectionPad2d(2),]
        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        # model6+=[norm_layer(512),]
        model6 += [nn.ReLU(True), ]
        # model6+=[nn.ReflectionPad2d(2),]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        # model6+=[norm_layer(512),]
        model6 += [nn.ReLU(True), ]
        # model6+=[nn.ReflectionPad2d(2),]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias), ]
        model6 += [nn.ReLU(True), ]
        model6 += [norm_layer(512), ]

        # Conv7
        # model47+=[nn.ReflectionPad2d(1),]
        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # model7+=[norm_layer(512),]
        model7 += [nn.ReLU(True), ]
        # model7+=[nn.ReflectionPad2d(1),]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # model7+=[norm_layer(512),]
        model7 += [nn.ReLU(True), ]
        # model7+=[nn.ReflectionPad2d(1),]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model7 += [nn.ReLU(True), ]
        model7 += [norm_layer(512), ]

        # Conv7
        model8up = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        # model3short8=[nn.ReflectionPad2d(1),]
        model3short8 = [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]

        # model47+=[norm_layer(256),]
        model8 = [nn.ReLU(True), ]
        # model8+=[nn.ReflectionPad2d(1),]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # model8+=[norm_layer(256),]
        model8 += [nn.ReLU(True), ]
        # model8+=[nn.ReflectionPad2d(1),]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model8 += [nn.ReLU(True), ]
        model8 += [norm_layer(256), ]

        # Conv9
        model9up = [nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]

        # model2short9=[nn.ReflectionPad2d(1),]
        model2short9 = [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # add the two feature maps above

        # model9=[norm_layer(128),]
        model9 = [nn.ReLU(True), ]
        # model9+=[nn.ReflectionPad2d(1),]
        model9 += [nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        model9 += [nn.ReLU(True), ]
        model9 += [norm_layer(128), ]

        # Conv10
        model10up = [nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias), ]

        # model1short10=[nn.ReflectionPad2d(1),]
        model1short10 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias), ]
        # add the two feature maps above

        # model10=[norm_layer(128),]
        model10 = [nn.ReLU(True), ]
        # model10+=[nn.ReflectionPad2d(1),]
        model10 += [nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias), ]
        model10 += [nn.LeakyReLU(negative_slope=.2), ]

        # regression output
        model_out = [nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias), ]
        if(use_tanh):
            model_out += [nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        #self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='nearest'), ])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1), ])

    def forward(self, real_L, fake_ab):
        conv1_2 = self.model1(torch.cat((real_L, fake_ab), dim=1))
        conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
        conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
        conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
        conv9_3 = self.model9(conv9_up)
        conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
        conv10_2 = self.model10(conv10_up)
        out_reg = self.model_out(conv10_2)

        return out_reg

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
