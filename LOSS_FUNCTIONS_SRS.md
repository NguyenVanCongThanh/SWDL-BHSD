# TÀI LIỆU TẢ CHỈ TIÊU KỸ THUẬT (SRS) VỀ CÁC HÀM LOSS (LOSS FUNCTIONS SPECIFICATION)

Tài liệu này mô tả chi tiết các hàm loss (hàm mất mát) được định nghĩa và sử dụng trong mã nguồn dự án Phân tích/Phân vùng hình ảnh y tế (SWDL). Các định nghĩa toán học, cài đặt mã nguồn và vai trò của từng hàm loss trong quá trình huấn luyện mô hình bán giám sát (Semi-supervised Learning) sẽ được làm rõ dưới đây.

---

## 1. Tổng Quan (Overview)
Trong bài toán phân vùng ảnh y khoa 3D (Medical Image Segmentation), đặc biệt là phân vùng bán giám sát (Semi-Supervised Segmentation), việc thiết kế hàm loss đóng vai trò quyết định. Mã nguồn sử dụng sự kết hợp của:
1. **Supervised Loss (Mất mát có giám sát):** Dành cho tập dữ liệu có nhãn đầy đủ (Labeled data), sử dụng Dice Loss và Cross-Entropy Loss.
2. **Consistency Loss (Mất mát tính nhất quán):** Dành cho dữ liệu không nhãn (Unlabeled data), giúp đảm bảo tính đồng nhất giữa các đầu ra dự đoán (ví dụ: Mean Squared Error - MSE, KL Divergence).
3. **Entropy Minimization (Giảm thiểu Entropy):** Thúc đẩy mô hình đưa ra dự đoán tự tin hơn trên dữ liệu không nhãn.
4. **Focal Loss:** Xử lý vấn đề mất cân bằng giữa các lớp (Class imbalance).

Các hàm loss này được định nghĩa tại [losses.py](file:///home/thanh/SWDL/code/utils/losses.py) và được gọi trong luồng huấn luyện tại [train_SWDL.py](file:///home/thanh/SWDL/code/BHSD2024/train_SWDL.py).

---

## 2. Chi Tiết Các Hàm Mất Mát (Detailed Loss Functions)

### 2.1. Dice Coefficient & Dice Loss
Họ hàm Dice dùng để đo lường mức độ tương đồng giữa vùng dự đoán và vùng nhãn chuẩn (Ground Truth).

#### a. Hàm `dice` (Hệ số Dice)
*   **Mục đích:** Tính toán hệ số tương đồng Dice (Dice Coefficient) cho ảnh phân vùng nhị phân (giá trị 0 hoặc 1).
*   **Công thức toán học:**
    $$\text{Dice} = \frac{2 \times |P \cap G| + \epsilon}{|P| + |G| + \epsilon}$$
    Trong đó $P$ là dự đoán, $G$ là ground truth, và $\epsilon$ (`smooth`) là hằng số cực tiểu để tránh lỗi chia cho 0.
*   **Ký hiệu hàm:** `dice(output_mask, target, smooth=1e-5)`

#### b. Hàm `dice_loss`
*   **Mục đích:** Tính Dice Loss dựa trên tổng bình phương của các phần tử.
*   **Công thức:**
    $$\text{Dice Loss} = 1 - \frac{2 \sum (p_i \cdot g_i) + \epsilon}{\sum p_i^2 + \sum g_i^2 + \epsilon}$$
*   **Ký hiệu hàm:** `dice_loss(score, target)`

#### c. Hàm `dice_loss1`
*   **Mục đích:** Tính Dice Loss phiên bản tuyến tính thông thường ở mẫu số.
*   **Công thức:**
    $$\text{Dice Loss 1} = 1 - \frac{2 \sum (p_i \cdot g_i) + \epsilon}{\sum p_i + \sum g_i + \epsilon}$$
*   **Ký hiệu hàm:** `dice_loss1(score, target)`

#### d. Lớp `DiceLoss` (PyTorch nn.Module)
*   **Mục đích:** Hỗ trợ tính Dice Loss đa lớp (Multi-class Segmentation) có tích hợp One-Hot Encoding cho nhãn đầu vào và hỗ trợ trọng số cho từng lớp (`weight`).
*   **Ký hiệu lớp:** `class DiceLoss(nn.Module)` với phương thức `forward(inputs, target, weight=None, softmax=False, oh_input=False)`

#### e. Hàm `dice_loss_masked`
*   **Mục đích:** Tính Dice Loss chỉ trên các khu vực được chỉ định bởi một mặt nạ (`mask`), loại bỏ các vùng không liên quan.
*   **Ký hiệu hàm:** `dice_loss_masked(score, target, mask=None)`

---

### 2.2. Intersection over Union (IoU) Loss
*   **Mục đích:** Đo lường độ chồng lấp Jaccard/IoU giữa dự đoán và nhãn. Thường dùng bổ trợ hoặc kết hợp cùng Dice Loss.
*   **Công thức toán học:**
    $$\text{IoU Loss} = 1 - \frac{|P \cap G| + \epsilon}{|P \cup G| + \epsilon} = 1 - \frac{\sum (p_i \cdot g_i) + \epsilon}{\sum p_i + \sum g_i - \sum (p_i \cdot g_i) + \epsilon}$$
*   **Cài đặt:**
    *   `iou_loss(score, target)`: Tính toán IoU Loss đơn lẻ.
    *   `dice_iou_loss(score, target)`: Bằng tổng của `dice_loss` + `iou_loss`.

---

### 2.3. Các hàm dựa trên Entropy (Entropy-based Losses)
Entropy đo lường mức độ không chắc chắn (uncertainty) của phân phối xác suất đầu ra.

#### a. Hàm `entropy_loss` & `entropy_minmization`
*   **Mục đích:** Giảm thiểu entropy để ép mô hình đưa ra các dự đoán rõ ràng và tự tin hơn (quyết đoán hơn ở các ranh giới).
*   **Công thức:**
    $$H(p) = - \frac{1}{\log(C)} \sum_{c=1}^{C} p_c \log(p_c + \epsilon)$$
*   **Cài đặt:**
    *   `entropy_loss(p, C=2)`: Trả về giá trị trung bình entropy trên toàn batch.
    *   `entropy_loss_map(p, C=2)` / `entropy_map(p)`: Trả về bản đồ entropy giữ nguyên kích thước không gian.

---

### 2.4. Tính nhất quán và Khoảng cách (Consistency & Distance Losses)

#### a. Hàm `mse_loss` & `symmetric_mse_loss`
*   **Mục đích:** Tính toán Mean Squared Error giữa hai dự đoán (như đầu ra của hai mô hình khác nhau hoặc cùng một mô hình dưới hai luồng nhiễu khác nhau).
*   **Công thức:**
    $$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (input1_i - input2_i)^2$$
*   **Cài đặt:**
    *   `mse_loss(input1, input2)`: Tính MSE thông thường.
    *   `symmetric_mse_loss(input1, input2)`: Giống MSE nhưng đảm bảo truyền gradient theo cả hai hướng thay vì chỉ một hướng.
    *   `softmax_mse_loss(input_logits, target_logits, sigmoid=False)`: Áp dụng Softmax (hoặc Sigmoid) lên cả hai đầu vào logits trước khi tính MSE.

#### b. Hàm `softmax_kl_loss` & `compute_kl_loss`
*   **Mục đích:** Sử dụng Kullback-Leibler Divergence để ép phân phối xác suất đầu ra của hai mạng khớp nhau.
*   **Cài đặt:**
    *   `softmax_kl_loss(input_logits, target_logits)`: Tính KL Divergence từ `target` sang `input` sau khi đi qua LogSoftmax và Softmax.
    *   `compute_kl_loss(p, q)`: Tính KL Divergence đối xứng hai chiều giữa `p` và `q`: $\text{Loss} = \frac{D_{KL}(p || q) + D_{KL}(q || p)}{2}$.

---

### 2.5. Focal Loss
*   **Mục đích:** Giải quyết vấn đề mất cân bằng nghiêm trọng giữa lớp nền (background) và lớp đối tượng cần phân vùng bằng cách tập trung phạt các mẫu "khó học".
*   **Công thức toán học:**
    $$\text{Focal Loss} = - \alpha_t (1 - p_t)^\gamma \log(p_t)$$
*   **Cài đặt:** `class FocalLoss(nn.Module)` với các siêu tham số $\gamma$ (`gamma`) và $\alpha$ (`alpha`).

---

## 3. Cách Sử Dụng Trong Quá Trình Huấn Luyện (Training Pipeline Integration)

Trong tệp [train_SWDL.py](file:///home/thanh/SWDL/code/BHSD2024/train_SWDL.py), hàm loss tổng hợp được tối ưu hóa như sau:

1.  **Supervised Loss (`loss_sup`):**
    Áp dụng trên dữ liệu có nhãn (`labeled_bs`). Là tổng hợp của:
    *   `loss_sup1`: Trung bình cộng Dice Loss trên từng lớp đối tượng:
        ```python
        loss_sup1 = losses.dice_loss(outputs_soft1[:labeled_bs, i+1, :, :, :], label_batch[:labeled_bs] == int(i+1))
        ```
    *   `loss_sup2`: Hàm Cross Entropy đa lớp tiêu chuẩn:
        ```python
        loss_sup2 = F.cross_entropy(outputs2[:labeled_bs, :, :, :, :], label_batch[:labeled_bs].long())
        ```
    *   Công thức: `loss_sup = loss_sup1 + loss_sup2`

2.  **Deep Supervision Loss (`loss_ds`):**
    Tính toán Dice Loss cho các nhánh đầu ra phụ (nếu có):
    ```python
    los1 = losses.dice_loss(out1_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
    # ... tính toán cho các nhánh khác và cộng dồn vào loss_ds
    ```

3.  **Consistency Loss (`loss_cons`):**
    Được tính trên cả dữ liệu có nhãn và không nhãn để duy trì tính nhất quán giữa hai mô hình/đầu ra khác nhau của kiến trúc mạng:
    ```python
    loss_cons = losses.mse_loss(outputs_soft1, outputs_soft2)
    ```

4.  **Tổng Hàm Mất Mát (Total Loss):**
    ```python
    loss = loss_sup + loss_ds + loss_cons
    ```
    Hàm loss tổng thể này liên tục cập nhật trọng số của mạng thông qua bộ scaler hỗn hợp (`scaler.scale(loss).backward()`).
