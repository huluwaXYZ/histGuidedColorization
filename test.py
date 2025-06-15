import skimage.metrics
import random
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from imageDataset import test_2imageDataset
from utils.util import *
from models.VGGNet import VGG19_feature_color_torchversion
from models.colorProviderNet import colorProviderNet_soft
from options.train_options import TrainOptions
from models import create_model
import torch
import torchvision.transforms as transforms
from utils import util

test_txt_path = './data/test_images.txt'

GLOBAL_SEED = 1
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)

set_seed(GLOBAL_SEED)

def testImagePairs(model_siggraph, vggnet, dataset_loader_test, opt):
    savepath = './save/test'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    with torch.no_grad():
        for index, data in enumerate(dataset_loader_test):
            imageX_L_norm = data[0].cuda()
            hist_tensor = data[1].cuda()
            ab_existOnImageX_tensor = data[2].cuda()
            referenceImage_ab_norm = data[3].cuda()

            # Lab to RGB
            zero_ab = torch.cat((torch.zeros_like(imageX_L_norm), torch.zeros_like(imageX_L_norm)), dim=1)
            imageX_RGB = tensor_lab2rgb(torch.cat((uncenter_l(imageX_L_norm), zero_ab), dim=1))

            colorProposal_ab = color_match(
                imageX_L_norm, imageX_RGB, hist_tensor, ab_existOnImageX_tensor, vggnet, model_siggraph.netColorProvider)

            data = {'real_L': imageX_L_norm,
                    'real_ab': referenceImage_ab_norm,
                    'fake_ab': colorProposal_ab}
            model_siggraph.set_input(data)
            model_siggraph.test(False)  # True means that losses will be computed
            visuals = util.get_subset_dict(model_siggraph.get_current_visuals(), to_visualize)
            output = visuals['fake_reg']
            out_ab = visuals['fake_ab']

            util.save_image(util.tensor2im(output), os.path.join(savepath,dataset_loader_test.dataset.imageDirs[index].split('/')[-1].split('.')[0]+'.png'))
            print('saved image:', os.path.join(savepath,dataset_loader_test.dataset.imageDirs[index].split('/')[-1].split('.')[0]+'.png'))

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.set_device(0)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    to_visualize = ['fake_reg','fake_ab']
    opt = TrainOptions().parse()

    colorProvider_net = colorProviderNet_soft(opt.batch_size, feature_channels_num=opt.vggFeatChannel_num, inter_channels_num=opt.featChannel_num)
    vggnet = VGG19_feature_color_torchversion()
    vggnet.load_state_dict(torch.load('./checkpoints/vgg19_conv.pth'))

    if len(opt.gpu_ids) > 0:
        assert(torch.cuda.is_available())
        #nonlocal_net.to(opt.gpu_ids[0])
        colorProvider_net = torch.nn.DataParallel(colorProvider_net, opt.gpu_ids).cuda()
        vggnet = torch.nn.DataParallel(vggnet, opt.gpu_ids).cuda()

    opt.netColorProvider = colorProvider_net

    model_histcol = create_model(opt)
    model_histcol.setup(opt)
    model_histcol.train()

    transform_colorize = transforms.Compose([transforms.Resize([opt.imageSize, opt.imageSize]), RGB2Lab(), ToTensor(), Normalize()])
    dataset_test = test_2imageDataset(test_txt_path, imageTransform=transform_colorize, binNum=256)
    dataset_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8,
                                     worker_init_fn=worker_init_fn, drop_last=True)

    model_histcol.netColorProvider.eval()
    model_histcol.netG.eval()
    vggnet.eval()

    testImagePairs(model_histcol, vggnet, dataset_loader_test, opt)

#python test.py --name experiment_soft3_256_0819 --load_model