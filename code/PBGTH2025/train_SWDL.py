import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
networks_dir = os.path.join(current_dir, '..', 'networks')
sys.path.append(networks_dir)
import SWDL
from SWDL import SWDL_Net

utils_dir = os.path.join(current_dir, '..', 'utils')
sys.path.append(utils_dir)
import ramps, losses
from losses import DiceLoss

dataloaders_dir = os.path.join(current_dir, '..', 'dataloaders')
sys.path.append(dataloaders_dir)
from PBGTH2025.PBGTH import PBGTH, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

from val_3D import val_all_case
from test_3D_util import test_all_case

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../../data/PBGTH_Dataset_RemoveSkull_resampled/dataSet')
parser.add_argument('--exp', type=str, default='PBGTH/SWDL')
parser.add_argument('--model', type=str, default='SWDL')
parser.add_argument('--max_iterations', type=int, default=15000)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--labeled_bs', type=int, default=2)
parser.add_argument('--base_lr', type=float, default=0.01)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--label_proportion', type=int, default=2)
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--temperature', type=float, default=0.05)
parser.add_argument('--ema_decay', type=float, default=0.99)
parser.add_argument('--consistency', type=float, default=1.0)
parser.add_argument('--consistency_rampup', type=float, default=40.0)
parser.add_argument('--pretrain', action='store_false')
parser.add_argument('--fold_th', type=str, default='fold_1')

args = parser.parse_args()

num_classes = 2
patch_size = (64, 64, 16)
train_data_path = args.root_path
snapshot_path = "../../model/" + args.exp + "_{:02d}p_{}/".format(args.label_proportion, args.fold_th)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
else:
    cudnn.benchmark = False
    cudnn.deterministic = True

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def create_model(ema=False):
    net = SWDL_Net(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True, has_residual=False)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

model = create_model()
model.eval()
input_tensor = torch.randn(1, 1, patch_size[2], patch_size[0], patch_size[1]).cuda()
outputs1, outputs2, masks, stage_out1, _ = model(input_tensor, [])
model.eval()

if args.pretrain:
    save_mode_path = os.path.join(snapshot_path, 'SWDL_best_model.pth')
    if os.path.exists(save_mode_path):
        state_dict = torch.load(save_mode_path)
        model.load_state_dict(state_dict)

db_train = PBGTH(base_dir=train_data_path,
                fold_th='/' + args.fold_th,
                split='train',
               transform=transforms.Compose([
                   RandomRotFlip(),
                   RandomCrop(patch_size),
                   ToTensor(),
               ]))

labelnum = round(len(db_train) * args.label_proportion / 100.)
labeled_idxs = list(range(labelnum))
unlabeled_idxs = list(range(labelnum, len(db_train)))
batch_sampler = TwoStreamBatchSampler(
    labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)

def worker_init_fn(worker_id):
    random.seed(args.seed + worker_id)

trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                         num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

model.train()
optimizer = optim.SGD(model.parameters(), lr=base_lr,
                      momentum=0.9, weight_decay=0.0001)
ce_loss = nn.CrossEntropyLoss(reduction='mean')
mse_loss = MSELoss()

writer = SummaryWriter(snapshot_path + '/log')
logging.info("{} itertations per epoch".format(len(trainloader)))

maxdice1 = 0.
iter_num = 0
max_epoch = max_iterations // len(trainloader) + 1
lr_ = base_lr
best_performance = 0.0
iterator = tqdm(range(max_epoch), ncols=70)

for epoch_num in iterator:
    time1 = time.time()
    for i_batch, sampled_batch in enumerate(trainloader):
        time2 = time.time()
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
        label_batch = label_batch == 1

        for num in range(3):
            model.train()
            if num == 0:
                outputs1, outputs2, masks, stage_out1, _ = model(volume_batch, [])
            else:
                outputs1, outputs2, masks, stage_out1, _ = model(volume_batch, en)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            en = []
            for idx in range(len(masks[0])):
                mask1 = masks[0][idx].detach()
                mask2 = masks[1][idx].detach()
                en.append(1e-3 * (mask1 - mask2))

            out5, out4, out3, out2, out1 = stage_out1[0], stage_out1[1], stage_out1[2], stage_out1[3], stage_out1[4]
            out1_soft = F.softmax(out1, dim=1)
            out2_soft = F.softmax(out2, dim=1)
            out3_soft = F.softmax(out3, dim=1)
            out4_soft = F.softmax(out4, dim=1)
            out5_soft = F.softmax(out5, dim=1)
            outputs_soft1 = F.softmax(outputs1, dim=1)
            outputs_soft2 = F.softmax(outputs2, dim=1)

            loss_sup1_sum = 0
            for i in range(num_classes-1):
                loss_sup1 = losses.dice_loss(outputs_soft1[:labeled_bs, i+1, :, :, :], label_batch[:labeled_bs] == int(i+1))
                loss_sup1_sum += loss_sup1

            loss_sup1 = loss_sup1_sum/(num_classes-1)
            loss_sup2 = F.cross_entropy(outputs2[:labeled_bs, :, :, :, :], label_batch[:labeled_bs].long())
            loss_sup = loss_sup1 + loss_sup2

            los1_sum = 0
            for i in range(num_classes-1):
                los1 = losses.dice_loss(out1_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
                los1_sum += los1

            los1 = los1_sum / (num_classes-1)
            los = 0.8 * los1
            loss_ds = los

            loss_cons = losses.mse_loss(outputs_soft1, outputs_soft2)
            loss = loss_sup + loss_ds + loss_cons

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        iter_num = iter_num + 1
        logging.info('iteration %d : loss : %f, loss_dice: %f, loss_ds: %f, loss_cons: %f' %
                    (iter_num, loss.item(), loss_sup.item(), loss_ds.item(), loss_cons.item()))

        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        if 0 < iter_num < max_iterations and iter_num % 50 == 0:
            model.eval()
            with torch.no_grad():
                avg_metric = val_all_case(
                    model, args.root_path, test_list="/" + args.fold_th + "/test.list", num_classes=num_classes,
                    patch_size=patch_size,
                    stride_xy=16, stride_z=8)
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)

                writer.add_scalar('info/val_dice_score',
                                  avg_metric[0, 0], iter_num)
                writer.add_scalar('info/val_hd95',
                                  avg_metric[0, 1], iter_num)
                logging.info('iteration %d : dice_score : %f hd95 : %f' % (
                    iter_num, avg_metric[0, 0].mean(), avg_metric[0, 1].mean()))
                model.train()

        if iter_num % 3000 == 0:
            save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
            torch.save(model.state_dict(), save_mode_path)

        if iter_num >= max_iterations:
            save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
            model.load_state_dict(torch.load(save_best))
            model.eval()
            test_save_path = "../../model/{}_{:02d}p_{}/Prediction".format(args.exp, args.label_proportion, args.fold_th)
            if not os.path.exists(test_save_path):
                os.makedirs(test_save_path)

            avg_metric = test_all_case(model, base_dir=args.root_path, method=args.model,
                                       test_list="/" + args.fold_th + "/test.list", num_classes=num_classes,
                                       patch_size=patch_size, stride_xy=8, stride_z=1,
                                       test_save_path=test_save_path)
            break

    if iter_num >= max_iterations:
        iterator.close()
        break
writer.close()
print("Training Finished!")