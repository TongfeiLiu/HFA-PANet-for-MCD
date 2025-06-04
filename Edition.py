import torch
import torch.nn as nn
import torch.utils.data.dataloader
import torchvision.transforms as transforms
from Dataset import LevirWhuGzDataset
from Dataset_HTCD import MT_HTCDDataset
from MASNet2 import FPANet_NoSaim
from tqdm import tqdm
from PIL import Image
import os
import argparse
from MMD_Loss import Dice_Loss, Hyper_Loss, DA_Loss, SoftDiceLoss
os.environ["CUDA_VISIBLE_DEVICES"]='0'

"""
Citing: 
Liu T, Pu Y, Lei T, et al. Hierarchical Feature Alignment-based Progressive Addition Network for Multimodal Change Detection[J]. Pattern Recognition, 2025, 162: 111355.
"""

parser = argparse.ArgumentParser(description='NS-FPANet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='HTCD', help='dataset')
parser.add_argument('--net', type=str, default='FPANet_NoSaim', help='net')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batchsize', type=float, default=8, help='batch size')
parser.add_argument('--pretrain', type=str, default='False', help='Is pretrain?')
parser.add_argument('--dataAug', type=str, default='False', help='Is DataAug?')
parser.add_argument('--lossType', type=str, default='BCE+DICE+0.1DA+0.1HML', help='Different lossType')
args = parser.parse_args()

DATASET = args.dataset
Net = args.net

transforms_set = transforms.Compose([
    transforms.ToTensor()
])
transforms_result = transforms.ToPILImage()

# MT-HTCD
if args.dataset == 'MTWHU':
    train_data = LevirWhuGzDataset(move='train',
                                   dataset=args.dataset,
                                   transform=transforms_set,
                                   isAug=args.dataAug)
    test_data = LevirWhuGzDataset(move='test',
                                  dataset=args.dataset,
                                  transform=transforms_set)
    args.lr = 0.0005
elif args.dataset == 'HTCD':
    data = MT_HTCDDataset(transform=transforms_set)
    torch.manual_seed(0)
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = torch.utils.data.random_split(data, [train_size, test_size], )
    args.lr = 0.0001


BATCH_SIZE = args.batchsize
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)
if args.net == 'FPANet_NoSaim':
    model = FPANet_NoSaim(pretrain=args.pretrain)

# model.load_state_dict(torch.load('HTCD_epoch_167_0.9638751839039584.pth'))

milestone = range(150, 200, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, gamma=0.9, milestones=milestone)

criterion = nn.BCEWithLogitsLoss()
SoftDice_Loss = SoftDiceLoss()

def confusion_matrix(true_value, output_data):
    image_size = true_value.shape[2]
    true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum = 0, 0, 0, 0
    true_value = torch.squeeze(true_value)
    output_data = torch.heaviside(torch.squeeze(output_data), torch.tensor([0], dtype=torch.float32, device='cuda'))
    batch_size = true_value.shape[0]
    for i in range(batch_size):
        union = torch.clamp(true_value[i] + output_data[i], 0, 1)
        intersection = true_value[i] * output_data[i]
        true_positive = int(intersection.sum())
        true_negative = image_size ** 2 - int(union.sum())
        false_positive = int((output_data[i] - intersection).sum())
        false_negative = int((true_value[i] - intersection).sum())
        true_positive_sum += true_positive
        true_negative_sum += true_negative
        false_positive_sum += false_positive
        false_negative_sum += false_negative

    return true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum


def save_visual_result(output_data, img_sequence):
    output_data = torch.heaviside(torch.squeeze(output_data), torch.tensor([0], dtype=torch.float32, device='cuda'))
    output_data = output_data.cpu().clone()
    batch_size = output_data.shape[0]
    for i in range(batch_size):
        image = transforms_result(output_data[i])
        img_sequence.append(image)
    return img_sequence


def save_attention_visual_result(output_data, img_sequence):
    output_data = torch.squeeze(output_data.cpu().clone())
    batch_size = output_data.shape[0]
    for i in range(batch_size):
        image = transforms_result(output_data[i])
        img_sequence.append(image)
    return img_sequence


def evaluate(tp, tn, fp, fn):
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    oa = ((tp + tn) / (tp + tn + fp + fn)) if (tp + tn + fp + fn) else 0
    recall = (tp / (tp + fn)) if (tp + fn) else 0
    precision = (tp / (tp + fp)) if (tp + fp) else 0
    f1 = (2 * ((precision * recall) / (precision + recall))) if (precision + recall) else 0
    false_alarm = (fp / (tn + fp)) if (tn + fp) else 0
    missing_alarm = (fn / (tp + fn)) if (tp + fn) else 0
    CIOU = (tp / (tp + fp + fn)) if (tp + fp + fn) else 0
    UCIOU = (tn / (tn + fp + fn)) if (tn + fp + fn) else 0
    MIOU = (CIOU + UCIOU) / 2

    return oa, recall, precision, f1, false_alarm, missing_alarm, CIOU, UCIOU, MIOU


def adjust_learning_rate(epoch_number, batch_idx, lr_decay_rate, base_lr, train_loader_arg, optimizer_arg):
    lr_adj = lr_decay_rate ** (epoch_number + float(batch_idx + 1) / len(train_loader_arg))
    for param_group in optimizer_arg.param_groups:
        param_group['lr'] = base_lr * lr_adj

def train(train_loader_arg, model_arg, criterion_arg, optimizer_arg, the_epoch, scheduler_arg):
    model_arg.cuda()
    model_arg.train()
    with tqdm(total=len(train_loader_arg), desc='Train Epoch #{}'.format(the_epoch + 1)) as t:
        for batch_idx, (img_1, img_2, label, masks) in tqdm(enumerate(train_loader_arg)):
            # adjust_learning_rate(the_epoch, batch_idx, 0.87, 0.01, train_loader_arg, optimizer_arg)
            img_1, img_2, label = img_1.cuda(), img_2.cuda(), label.cuda()
            output, preds, mmds = model_arg(img_1, img_2)
            level_masks = masks
            # Domain Alignment Loss
            DA_loss = DA_Loss(mmds, mode='mean')

            # Hyper Multi-Level Loss
            HML_loss = Hyper_Loss(preds, level_masks, mode='mean')

            # Foutput Fusion(bce+dice) Loss
            bce_Loss = criterion_arg(output, label)
            dice_Loss = Dice_Loss(output, label)

            if args.lossType == 'BCE':
                loss = bce_Loss
            elif args.lossType == 'BCE+0.1HML':
                loss = bce_Loss + 0.1 * HML_loss
            elif args.lossType == 'BCE+0.1DA':
                loss = bce_Loss + 0.1 * DA_loss
            elif args.lossType == 'BCE+0.1DA+0.1HML':
                loss = bce_Loss + 0.1 * DA_loss + 0.1 * HML_loss
            elif args.lossType == 'DICE':
                loss = SoftDice_Loss(output, label)
            elif args.lossType == 'DICE+0.1HML':
                loss = dice_Loss + 0.1 * HML_loss
            elif args.lossType == 'DICE+0.1DA':
                loss = dice_Loss + 0.1 * DA_loss
            elif args.lossType == 'DICE+0.1DA+0.1HML':
                loss = dice_Loss + 0.1 * DA_loss + 0.1 * HML_loss
            elif args.lossType == 'BCE+DICE':
                loss = bce_Loss + dice_Loss
            elif args.lossType == 'BCE+DICE+0.1HML':
                loss = bce_Loss + dice_Loss + 0.1 * HML_loss
            elif args.lossType == 'BCE+DICE+0.1DA':
                loss = bce_Loss + dice_Loss + 0.1 * DA_loss
            elif args.lossType == 'BCE+DICE+0.1DA+0.1HML':
                loss = bce_Loss + dice_Loss + 0.1 * DA_loss + 0.1 * HML_loss
            optimizer_arg.zero_grad()
            loss.backward()
            optimizer_arg.step()
            t.set_postfix({'lr': '%.5f' % optimizer_arg.param_groups[0]['lr'],
                           'loss': '%.4f' % loss.detach().cpu().data})
            t.update(1)
    scheduler_arg.step()


def test(test_loader_arg, model_arg, criterion_arg, the_epoch):
    images = []
    images_label = []
    test_loss = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    oa, recall, precision, f1, false_alarm, missing_alarm, ciou, uciou, miou = 0, 0, 0, 0, 0, 0, 0, 0, 0
    model_arg.cuda()
    model_arg.eval()
    with tqdm(total=len(test_loader_arg), desc='Test Epoch #{}'.format(the_epoch + 1)) as t:
        for batch_idx, (img_1, img_2, label, masks) in tqdm(enumerate(test_loader_arg)):
            img_1, img_2, label = img_1.cuda(), img_2.cuda(), label.cuda()
            output, preds, mmds = model_arg(img_1, img_2)
            test_loss += criterion_arg(output, label).item()
            tp_tmp, tn_tmp, fp_tmp, fn_tmp = confusion_matrix(label, output)
            images_label = save_visual_result(label, images_label)
            images = save_visual_result(output, images)
            tp += tp_tmp
            tn += tn_tmp
            fp += fp_tmp
            fn += fn_tmp
            if batch_idx >= 0:
                oa, recall, precision, f1, false_alarm, missing_alarm, ciou, uciou, miou = evaluate(tp, tn, fp, fn)
            t.set_postfix({'loss': '%.4f' % (test_loss / (batch_idx + 1)),
                           'acc': oa,
                           'f1': '%.4f' % f1,
                           'recall': '%.4f' % recall,
                           'precision': '%.4f' % precision,
                           'false alarm': '%.4f' % false_alarm,
                           'missing alarm': '%.4f' % missing_alarm,
                           'CIOU': '%.4f' % ciou,
                           'UCIOU': '%.4f' % uciou,
                           'MIOU': '%.4f' % miou})
            t.update(1)

    if (the_epoch + 1) >= 1:
        # torch.save(model_arg.state_dict(), "LEVIR_{}th.pth".format(the_epoch + 1))
        iou_sequence.append(f1)
        if not os.path.isdir('vision/' + Net + '/' + args.lossType):
            os.makedirs('vision/' + Net + '/' + args.lossType)
        f = open('vision/' + Net + '/' + args.lossType + '/' + DATASET + '_' + args.lossType + '_' + Net + '.txt', 'a')
        f.write("\"epoch\":\"" + "{}\"\n".format(the_epoch + 1))
        f.write("\"oa\":\"" + "{}\"\n".format(oa))
        f.write("\"f1\":\"" + "{}\"\n".format(f1))
        f.write("\"recall\":\"" + "{}\"\n".format(recall))
        f.write("\"precision\":\"" + "{}\"\n".format(precision))
        f.write("\"false alarm\":\"" + "{}\"\n".format(false_alarm))
        f.write("\"missing alarm\":\"" + "{}\"\n".format(missing_alarm))
        f.write("\"CIOU\":\"" + "{}\"\n".format(ciou))
        f.write("\"UCIOU\":\"" + "{}\"\n".format(uciou))
        f.write("\"MIOU\":\"" + "{}\"\n".format(miou))
        f.write("\"best f1 epoch\":\"" + "{}\"\n".format(iou_sequence.index(max(iou_sequence)) + 1))
        f.write("=====================================================================\n\n")
        f.close()
        print('max_f1:' + str(max(iou_sequence)) + ' epoch:' + str(iou_sequence.index(max(iou_sequence)) + 1) + '\n')

        if f1 == max(iou_sequence):
            for i in range(len(images)):
                result_label = images_label[i]
                result_image = images[i]
                if not os.path.isdir('vision/' + Net + '/' + args.lossType + '/result/' + DATASET):
                    os.makedirs('vision/' + Net + '/' + args.lossType + '/result/' + DATASET)
                Image.Image.save(result_image, 'vision/' + Net + '/' + args.lossType + '/result/' + DATASET + '/{}.png'.format(i))
                if not os.path.isdir('vision/' + Net + '/' + args.lossType + '/label/' + DATASET):
                    os.makedirs('vision/' + Net + '/' + args.lossType + '/label/' + DATASET)
                Image.Image.save(result_label, 'vision/' + Net + '/' + args.lossType + '/label/' + DATASET + '/{}.png'.format(i))
            torch.save(model_arg.state_dict(), "vision/{}/{}/{}_epoch_{}_{}.pth".format(Net, args.lossType, DATASET, the_epoch + 1, f1))

iou_sequence = []

for epoch in range(200):
    train(train_loader_arg=train_loader,
          model_arg=model,
          criterion_arg=criterion,
          optimizer_arg=optimizer,
          scheduler_arg=scheduler,
          the_epoch=epoch)
    if epoch >= 0:
        test(test_loader_arg=test_loader,
             model_arg=model,
             criterion_arg=criterion,
             the_epoch=epoch)
