import os
import cv2
import torch.utils.data
import random


# this function is for read image,the input is directory name
def read_directory(directory_name, label=False):
    array_of_img = []  # this if for store all of the image data
    # this loop is for read each image in this folder,directory_name is the folder name with images.
    files = os.listdir(r"/" + directory_name)
    files.sort(key=lambda x: int(x[0:-4]))
    for filename in files:
        # print(filename) #just for test
        # img is used to store the image data

        img = cv2.imread("/"+directory_name + "/" + filename)
        if label:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if len(img.shape) == 2:
            img = img.reshape(img.shape[0], img.shape[1], 1)
        array_of_img.append(img)
        # print(img)
        # print(array_of_img[0].shape)
    return array_of_img


dataset_LEVIR = 'CD_dataset/LEVIR'
dataset_WHU = 'CD_dataset/WHU'
dataset_PRCV = 'CD_dataset/SemiCD'
dataset_MTWHU = 'opt/data/private/Datasets/MCD/MCD_Scene/MTWHU'
dataset_MTOSCDOS = 'CD_dataset/MTOSCDOS'
dataset_MTOSCDSO = 'CD_dataset/MTOSCDSO'
dataset_CAU = 'CD_dataset/CAU'
dataset_train_1 = '/train/rgb'
dataset_train_2 = '/train/sar'
dataset_train_label = '/train/mask'
dataset_test_1 = '/test/rgb'
dataset_test_2 = '/test/sar'
dataset_test_label = '/test/mask'
MTOSCDOS_train_1 = '/train/opt_t1'
MTOSCDOS_train_2 = '/train/sar_t2'
MTOSCDOS_train_label = '/train/mask'
MTOSCDOS_test_1 = '/test/opt_t1'
MTOSCDOS_test_2 = '/test/sar_t2'
MTOSCDOS_test_label = '/test/mask'
MTOSCDSO_train_1 = '/train/sar_t1'
MTOSCDSO_train_2 = '/train/opt_t2'
MTOSCDSO_train_label = '/train/mask'
MTOSCDSO_test_1 = '/test/sar_t1'
MTOSCDSO_test_2 = '/test/opt_t2'
MTOSCDSO_test_label = '/test/mask'






class LevirWhuGzDataset(torch.utils.data.Dataset):
    def __init__(self, move='train', dataset='Gz', transform=None, isAug=False, isSwinT=False):
        super(LevirWhuGzDataset, self).__init__()
        seq_img_1 = []  # to pacify Pycharm
        seq_img_2 = []  # to pacify Pycharm
        seq_label = []  # to pacify Pycharm
        self.isaug = isAug
        self.isSwinT = isSwinT
        self.move = move
        if dataset == 'LEVIR':
            if move == 'train':
                seq_img_1 = read_directory(dataset_LEVIR + dataset_train_1)
                seq_img_2 = read_directory(dataset_LEVIR + dataset_train_2)
                seq_label = read_directory(dataset_LEVIR + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_LEVIR + dataset_test_1)
                seq_img_2 = read_directory(dataset_LEVIR + dataset_test_2)
                seq_label = read_directory(dataset_LEVIR + dataset_test_label, label=True)
        elif dataset == 'WHU':
            if move == 'train':
                seq_img_1 = read_directory(dataset_WHU + dataset_train_1)
                seq_img_2 = read_directory(dataset_WHU + dataset_train_2)
                seq_label = read_directory(dataset_WHU + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_WHU + dataset_test_1)
                seq_img_2 = read_directory(dataset_WHU + dataset_test_2)
                seq_label = read_directory(dataset_WHU + dataset_test_label, label=True)
        elif dataset == 'Gz':
            if move == 'train':
                seq_img_1 = read_directory(dataset_PRCV + dataset_train_1)
                seq_img_2 = read_directory(dataset_PRCV + dataset_train_2)
                seq_label = read_directory(dataset_PRCV + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_PRCV + dataset_test_1)
                seq_img_2 = read_directory(dataset_PRCV + dataset_test_2)
                seq_label = read_directory(dataset_PRCV + dataset_test_label, label=True)
        elif dataset == 'MTWHU':
            if move == 'train':
                seq_img_1 = read_directory(dataset_MTWHU + dataset_train_1)
                seq_img_2 = read_directory(dataset_MTWHU + dataset_train_2)
                seq_label = read_directory(dataset_MTWHU + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_MTWHU + dataset_test_1)
                seq_img_2 = read_directory(dataset_MTWHU + dataset_test_2)
                seq_label = read_directory(dataset_MTWHU + dataset_test_label, label=True)
        elif dataset == 'CAU':
            if move == 'train':
                seq_img_1 = read_directory(dataset_CAU + dataset_train_1)
                seq_img_2 = read_directory(dataset_CAU + dataset_train_2)
                seq_label = read_directory(dataset_CAU + dataset_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_CAU + dataset_test_1)
                seq_img_2 = read_directory(dataset_CAU + dataset_test_2)
                seq_label = read_directory(dataset_CAU + dataset_test_label, label=True)
        elif dataset == 'MTOSCDOS':
            if move == 'train':
                seq_img_1 = read_directory(dataset_MTOSCDOS + MTOSCDOS_train_1)
                seq_img_2 = read_directory(dataset_MTOSCDOS + MTOSCDOS_train_2)
                seq_label = read_directory(dataset_MTOSCDOS + MTOSCDOS_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_MTOSCDOS + MTOSCDOS_test_1)
                seq_img_2 = read_directory(dataset_MTOSCDOS + MTOSCDOS_test_2)
                seq_label = read_directory(dataset_MTOSCDOS + MTOSCDOS_test_label, label=True)
        elif dataset == 'MTOSCDSO':
            if move == 'train':
                seq_img_1 = read_directory(dataset_MTOSCDSO + MTOSCDSO_train_1)
                seq_img_2 = read_directory(dataset_MTOSCDSO + MTOSCDSO_train_2)
                seq_label = read_directory(dataset_MTOSCDSO + MTOSCDSO_train_label, label=True)
            elif move == 'test':
                seq_img_1 = read_directory(dataset_MTOSCDSO + MTOSCDSO_test_1)
                seq_img_2 = read_directory(dataset_MTOSCDSO + MTOSCDSO_test_2)
                seq_label = read_directory(dataset_MTOSCDSO + MTOSCDSO_test_label, label=True)
        self.seq_img_1 = seq_img_1
        self.seq_img_2 = seq_img_2
        self.seq_label = seq_label
        self.transform = transform

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        imgs_1 = self.seq_img_1[index]
        imgs_2 = self.seq_img_2[index]
        label = self.seq_label[index]

        # 随机进行数据增强，为2时不处理
        if self.isaug:
            flipCote = random.choice([-1, 0, 1, 2])
            if flipCote != 2:
                imgs_1 = self.augment(imgs_1, flipCote)
                imgs_2 = self.augment(imgs_2, flipCote)
                label = self.augment(label, flipCote)
                label = label.reshape(label.shape[0], label.shape[1], 1)
        h, w = label.shape[0], label.shape[1]
        if self.isSwinT:
            label_levels = [int(h / 8), int(h / 16), int(h / 32), int(h / 32)]
        else:
            label_levels = [int(h / 4), int(h / 4), int(h / 8), int(h / 16)]

        # level-1
        level_mask1 = cv2.resize(label, (label_levels[0], label_levels[0]))
        level_mask1 = level_mask1.reshape(level_mask1.shape[0], level_mask1.shape[1], 1)

        # level-2
        level_mask2 = cv2.resize(label, (label_levels[1], label_levels[1]))
        level_mask2 = level_mask2.reshape(level_mask2.shape[0], level_mask2.shape[1], 1)

        # level-3
        level_mask3 = cv2.resize(label, (label_levels[2], label_levels[2]))
        level_mask3 = level_mask3.reshape(level_mask3.shape[0], level_mask3.shape[1], 1)

        # level-4
        level_mask4 = cv2.resize(label, (label_levels[3], label_levels[3]))
        level_mask4 = level_mask4.reshape(level_mask4.shape[0], level_mask4.shape[1], 1)

        if self.move == 'train':
            train_masks = []
            if self.transform is not None:
                imgs_1 = self.transform(imgs_1)
                imgs_2 = self.transform(imgs_2)
                label = self.transform(label)
                level_mask1 = self.transform(level_mask1)
                level_mask2 = self.transform(level_mask2)
                level_mask3 = self.transform(level_mask3)
                level_mask4 = self.transform(level_mask4)
                train_masks.append(level_mask1)
                train_masks.append(level_mask2)
                train_masks.append(level_mask3)
                train_masks.append(level_mask4)
            return imgs_1, imgs_2, label, train_masks
        else:
            test_masks = []
            if self.transform is not None:
                imgs_1 = self.transform(imgs_1)
                imgs_2 = self.transform(imgs_2)
                label = self.transform(label)
                level_mask1 = self.transform(level_mask1)
                level_mask2 = self.transform(level_mask2)
                level_mask3 = self.transform(level_mask3)
                level_mask4 = self.transform(level_mask4)
                test_masks.append(level_mask1)
                test_masks.append(level_mask2)
                test_masks.append(level_mask3)
                test_masks.append(level_mask4)
            return imgs_1, imgs_2, label, test_masks

    def __len__(self):
        return len(self.seq_label)

