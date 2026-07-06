import os
import sys
import torch
import numpy as np
import SimpleITK as sitk
import h5py
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
networks_dir = os.path.join(current_dir, '..', 'networks')
sys.path.append(networks_dir)
from SWDL import SWDL_Net

from test_3D_util import test_single_case, calculate_metric_percase, cal_static

def resume_testing():
    # Các tham số cấu hình giống train_SWDL.py
    exp = 'BHSD/SWDL'
    model_name = 'SWDL'
    label_proportion = 5
    fold_th = 'fold_1'
    num_classes = 2
    patch_size = (96, 96, 32)
    stride_xy = 8
    stride_z = 1
    
    root_path = '/content/BHSD_Dataset_RemoveSkull_resampled/dataSet'
    if not os.path.exists(root_path):
        # Fallback cho local workspace nếu chạy offline
        root_path = '../../data/BHSD_Dataset_RemoveSkull_resampled/dataSet'

    snapshot_path = "../../model/" + exp + "_{:02d}p_{}/".format(label_proportion, fold_th)
    save_best = os.path.join(snapshot_path, '{}_best_model.pth'.format(model_name))
    test_save_path = os.path.join(snapshot_path, 'Prediction')
    file_path = os.path.join(test_save_path, "{}.txt".format(model_name))
    
    test_list_path = os.path.dirname(root_path) + '/{}/test.list'.format(fold_th)

    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)

    # 1. Đọc danh sách file test
    with open(test_list_path, 'r') as f:
        image_list = f.readlines()
    image_list = [root_path + "/{}".format(item.replace('\n', '').split(",")[0]) for item in image_list]

    # 2. Đọc các ID đã có sẵn trong file SWDL.txt để tránh tính toán lại
    completed_ids = set()
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]: # Bỏ qua header
                parts = line.strip().split(",")
                if parts and parts[0] and not parts[0].startswith("Mean metrics"):
                    completed_ids.add(parts[0])
    else:
        with open(file_path, "w") as f:
            f.write("ids, dice, ravd, hd, asd, accuracy, precision, jaccard, recall\n")
            f.flush()

    print(f"Đã tìm thấy {len(completed_ids)} cases đã được ghi vào {file_path}")

    # 3. Khởi tạo Model
    net = SWDL_Net(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True, has_residual=False)
    net = net.cuda()
    
    if os.path.exists(save_best):
        net.load_state_dict(torch.load(save_best, weights_only=False))
        print(f"Đã tải thành công trọng số mô hình từ {save_best}")
    else:
        print(f"LỖI: Không tìm thấy file trọng số tại {save_best}")
        return

    net.eval()

    metric_all = np.zeros([len(image_list), 8])
    total_metric = np.zeros((num_classes - 1, 8))
    
    # Mở file ghi kết quả với cơ chế ghi đè dòng/append linh hoạt
    # Chúng ta sẽ thu thập tất cả metrics của 39 cases vào metric_all
    for idx, image_path in enumerate(tqdm(image_list, desc="Testing cases")):
        ids = image_path.split("/")[-1].replace(".h5", "")
        
        # Đường dẫn các file nii.gz nếu đã được lưu
        pred_file = os.path.join(test_save_path, f"{ids}_pred.nii.gz")
        img_file = os.path.join(test_save_path, f"{ids}_img.nii.gz")
        lab_file = os.path.join(test_save_path, f"{ids}_lab.nii.gz")
        
        prediction = None
        label = None
        image = None
        
        # Trường hợp 1: Case này đã được ghi vào file SWDL.txt VÀ đã có file nii.gz
        if ids in completed_ids and os.path.exists(pred_file) and os.path.exists(lab_file):
            # Chỉ cần load lại label và prediction từ file để điền vào mảng thống kê cuối cùng
            pred_itk = sitk.ReadImage(pred_file)
            prediction = sitk.GetArrayFromImage(pred_itk)
            lab_itk = sitk.ReadImage(lab_file)
            label = sitk.GetArrayFromImage(lab_itk)
            metric = calculate_metric_percase(prediction == 1, label == 1)
        else:
            # Trường hợp 2: Chưa hoàn thành hoặc mất dòng trong file txt
            # Đọc file H5 gốc
            h5f = h5py.File(image_path, 'r')
            image = h5f['image'][:]
            label = h5f['label'][:]
            label = (label > 0).astype(np.uint8)
            
            # Nếu đã có file prediction nii.gz lưu trên ổ đĩa, load trực tiếp thay vì chạy model
            if os.path.exists(pred_file):
                pred_itk = sitk.ReadImage(pred_file)
                prediction = sitk.GetArrayFromImage(pred_itk)
            else:
                # Chạy mô hình dự đoán (Inference)
                prediction = test_single_case(
                    net, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
                
                # Lưu file nii.gz
                pred_itk = sitk.GetImageFromArray(prediction.astype(np.uint8))
                pred_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(pred_itk, pred_file)

                img_itk = sitk.GetImageFromArray(image)
                img_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(img_itk, img_file)

                lab_itk = sitk.GetImageFromArray(label.astype(np.uint8))
                lab_itk.SetSpacing((1.0, 1.0, 1.0))
                sitk.WriteImage(lab_itk, lab_file)
            
            # Tính toán metric
            metric = calculate_metric_percase(prediction == 1, label == 1)
            
            # Ghi dòng mới vào file SWDL.txt ngay lập tức và flush xuống đĩa
            with open(file_path, "a") as f_out:
                f_out.write("{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
                    ids, metric[0]*100, metric[1]*100, metric[2], metric[3], metric[4]*100, metric[5]*100, metric[6]*100, metric[7]*100))
                f_out.flush()
                os.fsync(f_out.fileno()) # Đảm bảo dữ liệu được ghi xuống đĩa cứng vật lý ngay lập tức
        
        metric_all[idx, :] = metric
        total_metric[0, :] += metric

    # 4. Ghi dòng giá trị trung bình (Mean metrics) và tính toán khoảng tin cậy thống kê ở cuối cùng
    # Xóa dòng Mean cũ nếu có để tránh trùng lặp
    clean_lines = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                if not (line.startswith("Mean metrics") or line.strip() == "" or len(line.split(",")) < 5):
                    clean_lines.append(line)
        with open(file_path, "w") as f:
            f.writelines(clean_lines)
            f.flush()

    with open(file_path, "a") as f:
        # Ghi dòng Mean
        f.write("Mean metrics,{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n".format(
            total_metric[0, 0]*100 / len(image_list), total_metric[0, 1]*100 / len(image_list),
            total_metric[0, 2] / len(image_list), total_metric[0, 3] / len(image_list),
            total_metric[0, 4]*100 / len(image_list), total_metric[0, 5]*100 / len(image_list),
            total_metric[0, 6]*100 / len(image_list), total_metric[0, 7]*100 / len(image_list)))
        
        # Tính toán bootstrap CI và kiểm định cho từng chỉ số
        names = ["dices", "ravds", "hds", "asds", "accuracys", "precisions", "jaccards", "recalls"]
        for i in range(8):
            metrics = metric_all[:, i]
            t_statistic, p_value, ci_lower, ci_upper = cal_static(metrics)
            # Đối với dice, ravd, accuracy, precision, jaccard, recall thì nhân 100
            if i in [0, 1, 4, 5, 6, 7]:
                f.write("{:.2f},{},{:.2f},{:.2f}\n".format(t_statistic, p_value, ci_lower*100, ci_upper*100))
            else:
                f.write("{:.2f},{},{:.2f},{:.2f}\n".format(t_statistic, p_value, ci_lower, ci_upper))
        f.flush()
        os.fsync(f.fileno())

    print("--- Đã hoàn thành quá trình Testing và cập nhật file SWDL.txt thành công! ---")

if __name__ == "__main__":
    resume_testing()
