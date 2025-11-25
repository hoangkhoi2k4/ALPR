# 🚗 Hướng Dẫn Training AI Nhận Diện Biển Số Xe Việt Nam

## 📋 Chuẩn Bị Trước Khi Đến Tiệm Net

### 1. Cài Đặt Môi Trường Trên Máy Cá Nhân

```bash
# Cài packages cơ bản
pip install -r requirements_training.txt

# Cài PyTorch với CUDA (tại tiệm net có GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Thu Thập Dataset

#### Nguồn Dataset Miễn Phí:

1. **Roboflow Universe**: https://universe.roboflow.com/
   - Tìm: "vietnam license plate" hoặc "vietnamese license plate"
   - Download format: **YOLOv8**
2. **Kaggle**: https://www.kaggle.com/datasets

   - Tìm: "vietnam license plate dataset"

3. **Tự Thu Thập**:
   - Chụp ảnh xe trên đường (tuân thủ quyền riêng tư)
   - Download ảnh từ nguồn hợp pháp
   - Cần: 1000-2000 ảnh train, 200-300 ảnh val

#### Công Cụ Label Ảnh:

- **LabelImg** (offline): https://github.com/HumanSignal/labelImg
- **Roboflow** (online, dễ dùng): https://roboflow.com/
- **CVAT** (chuyên nghiệp): https://www.cvat.ai/

### 3. Chuẩn Bị Dataset Trên Máy

```bash
# Tạo cấu trúc thư mục
python collect_dataset.py

# Đặt ảnh đã label vào:
# - dataset/raw_images/ (ảnh)
# - dataset/raw_labels/ (file .txt)

# Chia dataset tự động
python split_dataset.py
```

## 🎮 Tại Tiệm Net (Có GPU)

### 1. Kiểm Tra GPU

```python
python train_license_plate.py
# Sẽ hiển thị thông tin GPU nếu có
```

### 2. Training Model

#### Cấu Hình Khuyến Nghị:

**GPU Yếu (GTX 1060, GTX 1650):**

```python
from train_license_plate import train_model

train_model(
    epochs=50,
    batch=8,
    model_size='n',  # nano - nhỏ nhất
    imgsz=640
)
```

**GPU Trung Bình (RTX 3060, RTX 2070):**

```python
train_model(
    epochs=100,
    batch=16,
    model_size='s',  # small
    imgsz=640
)
```

**GPU Mạnh (RTX 3080, RTX 4090):**

```python
train_model(
    epochs=150,
    batch=32,
    model_size='m',  # medium
    imgsz=640
)
```

### 3. Thời Gian Training Ước Tính

| GPU      | Model   | Batch | Dataset  | Thời Gian/Epoch | 100 Epochs |
| -------- | ------- | ----- | -------- | --------------- | ---------- |
| GTX 1650 | YOLOv8n | 8     | 1000 ảnh | ~2-3 phút       | ~3-5 giờ   |
| RTX 3060 | YOLOv8s | 16    | 1000 ảnh | ~1-2 phút       | ~2-3 giờ   |
| RTX 3080 | YOLOv8m | 32    | 1000 ảnh | ~1 phút         | ~1.5-2 giờ |

### 4. Theo Dõi Training

Model sẽ tự động lưu:

- `runs/train/license_plate_vietnam/weights/best.pt` - Model tốt nhất
- `runs/train/license_plate_vietnam/weights/last.pt` - Model checkpoint cuối
- `runs/train/license_plate_vietnam/results.png` - Biểu đồ metrics

## 🔧 Xử Lý Lỗi Thường Gặp

### Lỗi: "CUDA out of memory"

**Giải pháp:**

```python
# Giảm batch size
train_model(batch=8)  # hoặc 4
```

### Lỗi: "No images found in dataset/images/train"

**Giải pháp:**

```bash
# Kiểm tra có ảnh trong thư mục
ls dataset/images/train

# Nếu không có, chạy lại
python collect_dataset.py
python split_dataset.py
```

### Lỗi: "No labels found"

**Giải pháp:**

- Đảm bảo mỗi ảnh có 1 file .txt tương ứng
- File .txt phải có format YOLO: `class x_center y_center width height`
- Các giá trị phải normalized (0-1)

### Training quá chậm

**Giải pháp:**

```python
# Giảm image size
train_model(imgsz=416)  # thay vì 640

# Tắt cache nếu thiếu RAM
# Sửa trong train_license_plate.py: cache=False
```

## 📊 Đánh Giá Model

### Validate Model

```python
from train_license_plate import validate_model

validate_model('runs/train/license_plate_vietnam/weights/best.pt')
```

### Test Thử Model

```python
from ultralytics import YOLO

model = YOLO('runs/train/license_plate_vietnam/weights/best.pt')

# Test 1 ảnh
results = model('test_image.jpg')
results[0].show()

# Test thư mục
results = model('test_images/')
```

## 💾 Sao Lưu Model

**Quan Trọng:** Copy model về máy trước khi rời tiệm net!

```bash
# Copy model tốt nhất
# Vào thư mục runs/train/license_plate_vietnam/weights/
# Copy file best.pt về USB hoặc upload lên Google Drive
```

## 🎯 Tiếp Theo

Sau khi có model (`best.pt`), bạn có thể:

1. **Tích hợp vào app**: Sửa `main.py` để dùng model mới
2. **Deploy lên server**: Dùng FastAPI, Flask
3. **Tối ưu cho mobile**: Export sang ONNX, TFLite
4. **Cải thiện model**: Train thêm với data mới

## 📞 Checklist Tại Tiệm Net

- [ ] Đã copy dataset vào máy
- [ ] Đã cài PyTorch + CUDA
- [ ] Đã kiểm tra GPU hoạt động
- [ ] Đã chạy train 1 epoch test thử
- [ ] Đang training (có thể để chạy, đi làm việc khác)
- [ ] Đã validate model
- [ ] **Đã copy model về USB/Drive trước khi về!**

## 🌟 Tips

1. **Training ban đêm**: Thuê tiệm net qua đêm (rẻ hơn), để train nhiều epochs
2. **Checkpoint**: Model tự động lưu mỗi 10 epochs, có thể dừng và tiếp tục sau
3. **Monitor**: Mở TensorBoard để xem live training: `tensorboard --logdir runs/train`
4. **Multiple runs**: Train nhiều config khác nhau, chọn model tốt nhất

---

**Chúc bạn training thành công! 🚀**
