from torch.utils.data import Dataset
from utils.util import *
from PIL import Image
from torchvision import transforms

class test_2imageDataset(Dataset):

    def __init__(self, txt_dir, imageTransform=transforms.ToTensor(), binNum=256):
        self.imageDirs, self.reference_image_dirs = get_feed_filePaths(txt_dir)
        self.binNum = binNum
        self.transform = imageTransform
        self.len = len(self.imageDirs)
        print(self.len)

    def __getitem__(self, index):
        imageX_RGB = Image.open(self.imageDirs[index]).convert("RGB")
        referenceImage_RGB = Image.open(self.reference_image_dirs[index]).convert("RGB")

        imageX_Lab_norm = self.transform(imageX_RGB)
        referenceImage_Lab_norm = self.transform(referenceImage_RGB)
        imageX_L_norm = imageX_Lab_norm[0:1, :, :]
        referenceImage_ab_norm = referenceImage_Lab_norm[1:3, :, :]
        # Exclude Zeros
        referenceImage_a_norm = referenceImage_Lab_norm[1, :, :].detach().numpy()
        referenceImage_b_norm = referenceImage_Lab_norm[2, :, :].detach().numpy()
        arr1 = referenceImage_a_norm.ravel()[np.flatnonzero(referenceImage_a_norm+referenceImage_b_norm)]
        arr2 = referenceImage_b_norm.ravel()[np.flatnonzero(referenceImage_a_norm+referenceImage_b_norm)]

        if (arr1.shape[0] != arr2.shape[0]):
            if arr2.shape[0] < arr1.shape[0]:
                arr2 = np.concatenate([arr2, np.array([0])])
            else:
                arr1 = np.concatenate([arr1, np.array([0])])

        hist, edges = np.histogramdd([arr1, arr2], bins=[self.binNum, self.binNum], range=((-1, 1), (-1, 1)), density=True)
        mask_hist = (hist > 0).astype(float)
        mask_hist = mask_hist[np.newaxis, :]

        hist_tensor = torch.from_numpy(hist[np.newaxis,:]).float()
        # select the ab values exist in the image
        a_allValues = np.array([2 * i / (self.binNum - 1) - 1 for i in range(self.binNum)])
        #
        a_allValues_2d = np.repeat(a_allValues[np.newaxis, :], repeats=self.binNum, axis=0)
        b_allValues_2d = a_allValues_2d.transpose(1, 0)
        ab_allValues = np.concatenate((a_allValues_2d[np.newaxis, :], b_allValues_2d[np.newaxis, :]), axis=0)
        ab_existOnImageX = ab_allValues * mask_hist
        ab_existOnImageX_tensor = torch.from_numpy(ab_existOnImageX).float()

        # input(L channel, hist, exist ab values), ab groundtruth of colorization
        return (imageX_L_norm, hist_tensor, ab_existOnImageX_tensor, referenceImage_ab_norm)

    def __len__(self):
        return self.len
