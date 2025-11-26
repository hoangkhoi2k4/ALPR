# 🚗 ALPR - Automatic License Plate Recognition (Vietnam)

Hệ thống nhận diện biển số xe Việt Nam tự động sử dụng YOLOv8.

## 📊 Kết Quả Training

### Model: YOLOv8s
- **Epochs**: 100
- **Training time**: 1.476 hours
- **Device**: NVIDIA GeForce RTX 3060 (12GB)

### Metrics
| Metric | BSD (Biển Dài) | BSV (Biển Vuông) | Overall |
|--------|----------------|------------------|---------|
| Precision | 0.992 | 0.977 | 0.985 |
| Recall | 0.995 | 0.981 | 0.988 |
| mAP50 | 0.995 | 0.993 | 0.994 |
| mAP50-95 | 0.907 | 0.923 | **0.915** |

### Performance
- **Inference Speed**: 2.6ms per image (~385 FPS)
- **Model Size**: 22.5MB
- **Parameters**: 11,126,358

## 📁 Cấu Trúc Dataset

```
dataset/
├── images/
│   ├── train/     # 3431 images
│   ├── val/       # 1145 images
│   └── test/      # (optional)
├── labels/
│   ├── train/     # YOLO format labels
│   └── val/
└── data.yaml      # Dataset config
```

### Format Label (YOLO)
```
<class_id> <x_center> <y_center> <width> <height>
```
- class_id: 0 = BSD (biển dài), 1 = BSV (biển vuông)
- Tọa độ normalized (0-1)

## 🚀 Quick Start

### 1. Cài Đặt

```bash
# Clone repository
git clone https://github.com/hoangkhoi2k4/ALPR.git
cd ALPR

# Cài dependencies
pip install -r requirements_training.txt

# Cài PyTorch với CUDA (nếu có GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Chuẩn Bị Dataset

```bash
# Thu thập và đặt ảnh vào dataset/raw_images/
# Label ảnh bằng LabelImg hoặc Roboflow

# Chia dataset tự động
python split_dataset.py
```

### 3. Training

```bash
# Training với GPU
python train_license_plate.py

# Hoặc tùy chỉnh
python -c "from train_license_plate import train_model; train_model(epochs=100, batch=16, model_size='s')"
```

### 4. Test/Inference

```bash
# Test với 1 ảnh
python -c "from ultralytics import YOLO; model = YOLO('runs/train/license_plate_vietnam5/weights/best.pt'); results = model.predict('path/to/image.jpg', save=True, conf=0.5)"

# Test với thư mục
python -c "from ultralytics import YOLO; model = YOLO('runs/train/license_plate_vietnam5/weights/best.pt'); results = model.predict('dataset/images/test/', save=True)"

# Webcam realtime
python -c "from ultralytics import YOLO; model = YOLO('runs/train/license_plate_vietnam5/weights/best.pt'); model.predict(source=0, show=True)"
```

## 📦 Files Quan Trọng

- `train_license_plate.py` - Script training chính
- `split_dataset.py` - Chia dataset train/val/test
- `collect_dataset.py` - Tạo cấu trúc thư mục
- `dataset/data.yaml` - Config dataset
- `runs/train/license_plate_vietnam5/weights/best.pt` - Model đã train

## 🎯 Classes

- **BSD** (class 0): Biển số dài (ô tô)
- **BSV** (class 1): Biển số vuông (xe máy)

## 📚 Tài Liệu

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - Hướng dẫn training chi tiết
- [quick_start.md](quick_start.md) - Hướng dẫn nhanh
- [SETUP_GITHUB.md](SETUP_GITHUB.md) - Setup GitHub

## 🔧 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8 (nếu training trên GPU)
- ultralytics (YOLOv8)
- opencv-python
- Pillow

## 📈 Training History

Xem chi tiết tại: `runs/train/license_plate_vietnam5/results.png`

## 👨‍💻 Author

- **Student**: Hoàng Khôi
- **Repository**: [hoangkhoi2k4/ALPR](https://github.com/hoangkhoi2k4/ALPR)

## 📝 License

Educational project for license plate recognition research.

## 🙏 Acknowledgments

- YOLOv8 by Ultralytics
- Dataset từ Roboflow Universe
- Support từ thầy giáo hướng dẫn
