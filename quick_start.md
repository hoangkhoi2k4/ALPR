# 🚀 Quick Start - Training Tại Tiệm Net RTX 3060

## ⚡ Checklist Nhanh

### 📋 TẠI NHÀ (15 phút)

```powershell
# 1. Push code lên GitHub
cd E:\clone\ALPR
git add .
git commit -m "Ready for training"
git push origin master

# 2. Upload dataset lên Google Drive
# Nén dataset/ thành zip và upload
# Lưu link share
```

### 🎮 TẠI TIỆM NET (10 phút setup + 2-3h training)

#### Lần Đầu Tiên:

```powershell
# 1. Clone repo
cd D:\workspace
git clone https://github.com/YOUR_USERNAME/ALPR-Training.git
cd ALPR-Training

# 2. Cài môi trường (5 phút)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python pillow numpy pandas matplotlib pyyaml tqdm

# 3. Download dataset từ Drive (3 phút)
# Giải nén vào thư mục dataset/

# 4. Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 5. Training (2-3 giờ)
python
>>> from train_license_plate import train_model
>>> train_model(epochs=100, batch=16, model_size='s')
```

#### Lần Sau (Nếu Đã Setup):

```powershell
# 1. Update code
cd D:\workspace\ALPR-Training
git pull origin master

# 2. Training ngay
python
>>> from train_license_plate import train_model
>>> train_model(epochs=100, batch=16, model_size='s')
```

### 💾 SAU KHI TRAIN (5 phút)

```powershell
# Model tại: runs/train/license_plate_vietnam/weights/best.pt

# Upload lên Drive
# Hoặc push Git:
git add runs/train/*/weights/best.pt
git commit -m "Trained model - RTX 3060"
git push origin master
```

---

## ⚙️ Config Cho RTX 3060 (12GB VRAM)

### Option 1: Cân Bằng (Khuyến nghị)

```python
train_model(
    epochs=100,
    batch=16,
    model_size='s',  # Small - nhanh, chính xác tốt
    imgsz=640
)
# Thời gian: ~2h với 1000 ảnh
```

### Option 2: Nhanh

```python
train_model(
    epochs=80,
    batch=24,
    model_size='n',  # Nano - cực nhanh
    imgsz=640
)
# Thời gian: ~1h với 1000 ảnh
```

### Option 3: Chính Xác

```python
train_model(
    epochs=150,
    batch=12,
    model_size='m',  # Medium - chính xác cao
    imgsz=640
)
# Thời gian: ~4h với 1000 ảnh
```

---

## 📊 Dataset Requirements

### Tối Thiểu:

- Train: 500+ ảnh
- Val: 100+ ảnh

### Khuyến Nghị:

- Train: 1500-2000 ảnh
- Val: 300-400 ảnh
- Test: 200 ảnh

### Nguồn Dataset:

1. **Roboflow**: https://universe.roboflow.com/

   - Tìm: "vietnam license plate"
   - Format: YOLOv8

2. **Tự thu thập + Label**:
   - LabelImg: https://github.com/HumanSignal/labelImg
   - Roboflow online tool

---

## 🐛 Xử Lý Lỗi Nhanh

### "CUDA out of memory"

```python
# Giảm batch size
train_model(batch=8)  # từ 16 xuống 8
```

### "No images found"

```powershell
# Kiểm tra dataset
ls dataset\images\train
ls dataset\labels\train
# Phải có file .jpg và .txt tương ứng
```

### "Git push failed"

```powershell
git pull origin master --rebase
git push origin master
```

### GPU không hoạt động

```powershell
# Cài lại PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 📱 Contact & Backup

### Trước Khi Rời Tiệm:

- [ ] Upload best.pt lên Drive
- [ ] Push code lên Git (nếu có thay đổi)
- [ ] Screenshot kết quả training
- [ ] Note lại metrics (mAP, precision, recall)

### Nếu Có Vấn Đề:

1. Copy toàn bộ folder runs/ vào USB (backup)
2. Screenshot lỗi
3. Commit code trước khi sửa:
   ```powershell
   git add .
   git commit -m "Before fixing issue"
   git push
   ```

---

## 🎯 Kết Quả Mong Đợi

Sau khi training 100 epochs với 1500 ảnh:

- **mAP50**: > 0.85 (tốt)
- **mAP50-95**: > 0.60 (OK)
- **Precision**: > 0.80
- **Recall**: > 0.75

Nếu thấp hơn → Cần thêm data hoặc train thêm epochs!

---

## 💰 Chi Phí Ước Tính

**Tiệm net có RTX 3060:**

- 3h training ≈ 15-30k VNĐ (tùy tiệm)
- Upload/download: miễn phí (WiFi tiệm)

**Tổng:** ~20-30k/lần training

---

## ⏱️ Timeline Hoàn Chỉnh

| Thời gian    | Công việc                           |
| ------------ | ----------------------------------- |
| **Tại nhà**  |
| 5 phút       | Push code lên Git                   |
| 10 phút      | Upload dataset lên Drive            |
| **Tại tiệm** |
| 5 phút       | Clone repo + setup                  |
| 3 phút       | Download dataset                    |
| 2 phút       | Check GPU & test                    |
| **2-3 giờ**  | **Training (có thể làm việc khác)** |
| 5 phút       | Upload model về Drive               |
| **Về nhà**   |
| 3 phút       | Download model                      |
| 5 phút       | Test model                          |

**Tổng thời gian:** ~3h active work, 2-3h passive (training chạy background)

---

Sẵn sàng chưa? Bắt đầu từ bước 1! 🚀
