import os
import cv2
import torch.utils.data
import numpy as np
# import torchvision.transforms as transforms



def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    return img.float().div(255)


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

data_MT_HTCD = '/opt/data/private/Datasets/MCD/MCD_Scene/MT-HTCD'

# transforms_set = transforms.Compose([transforms.ToTensor()])

class MT_HTCDDataset(torch.utils.data.Dataset):
    # img1-sat img2-uav
    def __init__(self, transform=None):
        super(MT_HTCDDataset, self).__init__()

        # ls_pick_images:选中的大图号（int）的list,方便划分数据集之用
        self.dir = data_MT_HTCD
        self.images = os.listdir(os.path.join(self.dir,'uav'))
        # self.sat_mean = np.array([66, 71, 74], np.uint8)
        # self.uav_mean = np.array([73, 81, 79], np.uint8)
        self.transform = transform


    def __getitem__(self, idx):
        # img1-sat img2-uav
        filename = self.images[idx]
        sat_file = os.path.join(self.dir, 'sat', filename)

        uav_file = os.path.join(self.dir, 'uav', filename)

        label_file = os.path.join(self.dir, 'label', filename)

        # print('----------', 'name:', filename,'----------')

        img1 = cv2.imread(sat_file)
        # img1 -= self.sat_mean
        # print('len img shape:',len(img1.shape))
        if (img1 is None):
            print(idx)
            print(sat_file)
        img_size = img1.shape[:2]
        # print('img_size:',img_size)
        # print('img1 size:',img1.shape)

        # img1 = img1.transpose((2, 0, 1)).astype(np.float32) / 128

        img2 = cv2.imread(uav_file)
        # img2 = cv2.resize(img2, (2048,2048)).astype(np.int)
        if img2 is None:
            print(idx)
            print(uav_file)
        else:
            img2 = cv2.resize(img2,img_size)
        # img2 -= self.uav_mean
        # print('img2 size:',img2.shape)
        # img2 = img2.transpose((2, 0, 1)).astype(np.float32) / 128

        # lbl = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_file, cv2.IMREAD_UNCHANGED)
        # print(np.max(lbl))
        # lbl = cv2.cvtColor(lbl, cv2.COLOR_BGR2GRAY)

        label = cv2.resize(label, img_size)
        masks = []
        h, w = label.shape[0], label.shape[1]

        label_levels = [int(h / 4), int(h / 4), int(h / 8), int(h / 16)]

        # level-1
        level_mask1 = cv2.resize(label, (label_levels[0], label_levels[0]))
        level_mask1 = level_mask1.reshape(level_mask1.shape[0], level_mask1.shape[1], 1)

        # level-2
        level_mask2 = cv2.resize(label, (label_levels[1], label_levels[1]))
        level_mask2 = level_mask1.reshape(level_mask2.shape[0], level_mask2.shape[1], 1)

        # level-3
        level_mask3 = cv2.resize(label, (label_levels[2], label_levels[2]))
        level_mask3 = level_mask3.reshape(level_mask3.shape[0], level_mask3.shape[1], 1)

        # level-4
        level_mask4 = cv2.resize(label, (label_levels[3], label_levels[3]))
        level_mask4 = level_mask4.reshape(level_mask4.shape[0], level_mask4.shape[1], 1)
        level_mask1 = self.transform(level_mask1) * 255
        level_mask2 = self.transform(level_mask2) * 255
        level_mask3 = self.transform(level_mask3) * 255
        level_mask4 = self.transform(level_mask4) * 255
        masks.append(level_mask1)
        masks.append(level_mask2)
        masks.append(level_mask3)
        masks.append(level_mask4)
        # lbl = np.asarray(lbl)
        # print('label size:', lbl.shape)
        img1 = toTensor(img1)
        img2 = toTensor(img2)
        lbl = self.transform(label) * 255
        # img1, img2, lbl = self.transform(img1), self.transform(img2), self.transform(lbl)
        return img1, img2, lbl, masks


    def __len__(self):
        return len(self.images)


