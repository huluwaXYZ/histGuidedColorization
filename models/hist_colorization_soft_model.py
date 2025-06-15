import torch
from collections import OrderedDict
from utils import util
from models.base_model import BaseModel
from models import networks

class histColorizationSoftModel(BaseModel):
    def name(self):
        return 'histColorizationSoftModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.netColorProvider = opt.netColorProvider
        self.loss_names = []

        self.loss_names += ['G_L1_reg', 'G_huber_grad']

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks

        self.model_names = ['G', 'ColorProvider']

        # load/define networks
        num_in = opt.input_nc + opt.featChannel_num
        self.netG = networks.define_G(num_in, opt.output_nc, opt.norm, opt.init_type, self.gpu_ids, use_tanh=True)

        if self.isTrain:
            self.criterionL1 = networks.L1Loss()
            self.criterionHuber = networks.HuberLoss(delta=1. / opt.ab_norm)
            self.criterionGradient = networks.GradientLoss(delta=1. / opt.ab_norm, channel_num=2)
            self.criterionHistogram = networks.HistogramLoss_2channel(bin_num=opt.bin_num, delta=1. / opt.ab_norm, imSize=opt.imageSize)

            # initialize optimizers
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam([{'params':self.netG.parameters()}, {'params':self.netColorProvider.parameters()},],
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        self.real_L = input['real_L'].to(self.device)
        self.real_ab = input['real_ab'].to(self.device)
        self.fake_ab = input['fake_ab'].to(self.device)

    def forward(self):
        self.fake_B_reg = self.netG(self.real_L, self.fake_ab)

    def compute_losses_G(self):
        self.loss_G_L1_reg = torch.mean(self.criterionHuber(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                              self.real_ab.type(torch.cuda.FloatTensor)))
        self.loss_G_huber_grad = torch.mean(self.criterionGradient(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                              self.real_ab.type(torch.cuda.FloatTensor)))
        self.loss_G_hist = torch.mean(self.criterionHistogram(self.fake_B_reg.type(torch.cuda.FloatTensor),
                                                              self.real_ab.type(torch.cuda.FloatTensor)))
        self.loss_G = self.loss_G_L1_reg + self.loss_G_huber_grad + self.loss_G_hist

    def backward_G(self):
        self.compute_losses_G()
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        # update G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()
        visual_ret['fake_reg'] = util.lab2rgb(torch.cat((self.real_L.type(torch.cuda.FloatTensor), self.fake_B_reg.type(torch.cuda.FloatTensor)), dim=1), self.opt)
        visual_ret['fake_ab'] = self.fake_B_reg
        return visual_ret

