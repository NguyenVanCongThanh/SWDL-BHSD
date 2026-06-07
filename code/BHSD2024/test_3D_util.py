import math

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from skimage.measure import label
from tqdm import tqdm
import os
import numpy as np
from scipy import stats
from scipy.stats import norm

def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2, w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2, h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2, d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # 调整维度顺序：从 (width, height, depth) 到 (depth, height, width)
                test_patch = np.transpose(test_patch, (2, 1, 0))  # 将 depth 放到第 0 维
                test_patch = np.expand_dims(np.expand_dims(
                    test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1, _, _, _, _ = net(test_patch, [])
                    # y1 = net(test_patch)
                    # ensemble
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                # 调整维度顺序：从 (channel, depth, height, width)到 (channel, width, height, depth)
                y = np.transpose(y, (0, 3, 2, 1))  # 将 depth 放到第 0 维
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                    = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1
    score_map = score_map/np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,
                              hl_pad:hl_pad+h, dl_pad:dl_pad+d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad+h, dl_pad:dl_pad+d]
    return label_map


def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        return np.zeros(2)

# 2. 计算 95% 置信区间 (Bootstrap 方法)
def cal_static(metrics):
    # 1. 计算 p 值
    t_statistic, p_value = stats.ttest_1samp(metrics, popmean=0.5, alternative='greater')
    print(f"t 统计量: {t_statistic}")
    print(f"p 值: {p_value}")

    # 2. 计算 95% 置信区间 (Bootstrap 方法)
    lower, upper = bootstrap_ci(metrics)
    print(f"95% 置信区间 (Bootstrap): [{lower}, {upper}]")

    # 3. 计算 95% 置信区间 (正态分布假设)
    mean = np.mean(metrics)
    std = np.std(metrics, ddof=1)  # 使用样本标准差
    n = len(metrics)

    z = norm.ppf(0.975)  # 95% 置信区间的 Z 值
    margin_of_error = z * (std / np.sqrt(n))
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    print(f"95% 置信区间 (正态分布假设): [{ci_lower}, {ci_upper}]")

    return t_statistic, p_value, ci_lower, ci_upper
def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)
    return lower, upper

def test_all_case(net, base_dir, method="unet_3D", test_list="full_test.list", num_classes=4, patch_size=(48, 160, 160), stride_xy=32, stride_z=24, test_save_path=None):
    parent_dir_path = os.path.dirname(base_dir)
    with open(parent_dir_path + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes-1, 8))
    print("Testing begin")
    file_path = test_save_path + "/{}.txt".format(method)
    # 检查文件是否存在
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("ids, dice, ravd, hd, asd, accuracy, precision, jaccard, recall\n")
    metric_all = np.zeros([len(image_list),8])
    cont = 0
    with open(file_path, "a") as f:
        for image_path in tqdm(image_list):
            ids = image_path.split("/")[-1].replace(".h5", "")
            h5f = h5py.File(image_path, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            label = (label > 0).astype(np.uint8)
            prediction = test_single_case(
                net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
            metric = calculate_metric_percase(prediction == 1, label == 1)
            total_metric[0, :] += metric
            metric_all[cont,:] = metric
            cont += 1
            f.writelines("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                ids, metric[0]*100, metric[1]*100, metric[2], metric[3], metric[4]*100, metric[5]*100, metric[6]*100, metric[7]*100))

            pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
            pred_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(pred_itk, test_save_path +
                            "/{}_pred.nii.gz".format(ids))

            img_itk = sitk.GetImageFromArray(image)
            img_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(img_itk, test_save_path +
                            "/{}_img.nii.gz".format(ids))

            lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
            lab_itk.SetSpacing((1.0, 1.0, 1.0))
            sitk.WriteImage(lab_itk, test_save_path +
                            "/{}_lab.nii.gz".format(ids))
        f.writelines("Mean metrics,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(total_metric[0, 0]*100 / len(image_list), total_metric[0, 1]*100 / len(
            image_list), total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list), total_metric[0, 4]*100 / len(image_list), total_metric[0, 5]*100 / len(
            image_list), total_metric[0, 6]*100 / len(image_list), total_metric[0, 7]*100 / len(image_list)))
    f.close()

    dices = metric_all[:, 0]
    t_statistic, p_value, ci_lower, ci_upper = cal_static(dices)
    with open(file_path, "a") as f:
        f.writelines("{:.2f},{},{:.2f},{:.2f}\n".format(
            t_statistic, p_value, ci_lower*100, ci_upper*100))

    ravds = metric_all[:, 1]
    t_statistic, p_value, ci_lower, ci_upper = cal_static(ravds)
    with open(file_path, "a") as f:
        f.writelines("{:.2f},{},{:.2f},{:.2f}\n".format(
            t_statistic, p_value, ci_lower*100, ci_upper*100))

    hds = metric_all[:, 2]
    t_statistic, p_value, ci_lower, ci_upper = cal_static(hds)
    with open(file_path, "a") as f:
        f.writelines("{:.2f},{},{:.2f},{:.2f}\n".format(
            t_statistic, p_value, ci_lower, ci_upper))

    asds = metric_all[:, 3]
    t_statistic, p_value, ci_lower, ci_upper = cal_static(asds)
    with open(file_path, "a") as f:
        f.writelines("{:.2f},{},{:.2f},{:.2f}\n".format(
            t_statistic, p_value, ci_lower, ci_upper))

    accuracys = metric_all[:, 4]
    t_statistic, p_value, ci_lower, ci_upper = cal_static(accuracys)
    with open(file_path, "a") as f:
        f.writelines("{:.2f},{},{:.2f},{:.2f}\n".format(
            t_statistic, p_value, ci_lower*100, ci_upper*100))

    precisions = metric_all[:, 5]
    t_statistic, p_value, ci_lower, ci_upper = cal_static(precisions)
    with open(file_path, "a") as f:
        f.writelines("{:.2f},{},{:.2f},{:.2f}\n".format(
            t_statistic, p_value, ci_lower*100, ci_upper*100))

    jaccards = metric_all[:, 6]
    t_statistic, p_value, ci_lower, ci_upper = cal_static(jaccards)
    with open(file_path, "a") as f:
        f.writelines("{:.2f},{},{:.2f},{:.2f}\n".format(
            t_statistic, p_value, ci_lower*100, ci_upper*100))

    recalls = metric_all[:, 7]
    t_statistic, p_value, ci_lower, ci_upper = cal_static(recalls)
    with open(file_path, "a") as f:
        f.writelines("{:.2f},{},{:.2f},{:.2f}".format(
            t_statistic, p_value, ci_lower*100, ci_upper*100))

    f.close()

    print("Testing end")
    return total_metric / len(image_list)




def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / \
            (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


# def calculate_metric_percase(pred, gt):
#     dice = metric.binary.dc(pred, gt)
#     ravd = abs(metric.binary.ravd(pred, gt))
#     hd = metric.binary.hd95(pred, gt)
#     asd = metric.binary.asd(pred, gt)
#     return np.array([dice, ravd, hd, asd])


def calculate_metric_percase(pred, gt):
    """
    计算多种分割指标。

    参数:
    pred (numpy array): 预测结果（二值掩码）。
    gt (numpy array): 真实标签（二值掩码）。

    返回:
    numpy array: 包含 Dice, RAVD, HD95, ASD, IoU, Accuracy, Precision, Jaccard, Recall 的数组。
    """
    # 确保输入是二值掩码
    if np.sum(pred) == 0:
        print("警告：pred 没有前景像素，手动添加一个前景像素。")
        pred[0, 0, 0] = 1  # 将第一个像素设置为 1
        gt[0, 0, 0] = 1  # 将第一个像素设置为 1

        # 检查 gt 是否包含前景像素
    if np.sum(gt) == 0:
        print("警告：gt 没有前景像素，手动添加一个前景像素。")
        pred[0, 0, 0] = 1  # 将第一个像素设置为 1
        gt[0, 0, 0] = 1  # 将第一个像素设置为 1

    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # 计算 Dice Coefficient
    dice = metric.binary.dc(pred, gt)

    # 计算 Relative Absolute Volume Difference
    ravd = abs(metric.binary.ravd(pred, gt))

    # 计算 95th Percentile Hausdorff Distance
    hd = metric.binary.hd95(pred, gt)

    # 计算 Average Symmetric Surface Distance
    asd = metric.binary.asd(pred, gt)

    # 计算 Intersection over Union (IoU)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    iou = intersection / union if union != 0 else 0.0

    # 计算 Accuracy
    accuracy = np.sum(pred == gt) / gt.size

    # 计算 Precision
    true_positive = np.logical_and(pred, gt).sum()
    false_positive = np.logical_and(pred, np.logical_not(gt)).sum()
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0.0

    # 计算 Jaccard (等同于 IoU)
    jaccard = iou

    # 计算 Recall
    false_negative = np.logical_and(np.logical_not(pred), gt).sum()
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0.0

    # 返回所有指标
    return np.array([dice, ravd, hd, asd, accuracy, precision, jaccard, recall])