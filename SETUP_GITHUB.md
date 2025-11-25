# 🔧 Setup GitHub Workflow cho Training AI

## 📋 Chuẩn Bị

### 1. Cài Git (nếu chưa có)

```powershell
# Kiểm tra
git --version

# Nếu chưa có, download tại:
# https://git-scm.com/download/win
```

### 2. Tạo GitHub Account

- Đăng ký miễn phí tại: https://github.com/
- Verify email

---

## 🏠 Bước 1: TẠI NHÀ (Chuẩn Bị Code)

### A. Khởi tạo Git

```powershell
cd E:\clone\ALPR

# Init git (nếu chưa có)
git init

# Config thông tin (lần đầu)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### B. Setup .gitignore

```powershell
# Tự động tạo .gitignore
python git_workflow.py
# Chọn: 1. Setup Git

# Hoặc thủ công: tạo file .gitignore với nội dung sau
```

**Nội dung .gitignore quan trọng:**

```
# Dataset - KHÔNG push lên Git (quá nặng!)
dataset/images/
dataset/labels/
*.jpg
*.jpeg
*.png

# Model - Dùng Git LFS hoặc upload riêng
*.pt
*.pth
runs/

# Python
__pycache__/
*.pyc
```

### C. Tạo Repo Trên GitHub

1. Vào: https://github.com/new
2. Đặt tên: `ALPR-Training`
3. Chọn: **Private** (để bảo mật)
4. **KHÔNG** tích "Add README" (vì đã có code)
5. Create repository

### D. Link Repo & Push

```powershell
# Thêm remote (thay YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ALPR-Training.git

# Add & commit code
git add .
git commit -m "Initial commit: training scripts"

# Push lên GitHub
git branch -M master
git push -u origin master
```

**Lần đầu push sẽ yêu cầu đăng nhập GitHub!**

---

## 📤 Bước 2: Upload Dataset (Riêng)

**QUAN TRỌNG:** Dataset KHÔNG push lên Git vì quá nặng!

### Cách 1: Google Drive (Khuyến nghị)

```
1. Nén dataset thành file .zip:
   • dataset/images/train/
   • dataset/images/val/
   • dataset/labels/train/
   • dataset/labels/val/

2. Upload lên Google Drive

3. Share link (Anyone with the link)

4. Note link này để download tại tiệm net
```

### Cách 2: MEGA.nz

- Upload dataset folder
- Share link
- Download tại tiệm net

### Cách 3: Đem USB vật lý

- Copy dataset vào USB
- Mang đến tiệm net
- Copy vào máy tiệm

---

## 🎮 Bước 3: TẠI TIỆM NET (RTX 3060)

### A. Clone Repo

```powershell
# Tạo thư mục làm việc
mkdir D:\workspace
cd D:\workspace

# Clone code từ GitHub
git clone https://github.com/YOUR_USERNAME/ALPR-Training.git
cd ALPR-Training
```

### B. Cài Môi Trường

```powershell
# Cài PyTorch với CUDA 11.8 (cho RTX 3060)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Cài packages
pip install -r requirements_training.txt

# Kiểm tra GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

**Kết quả mong muốn:**

```
CUDA: True, GPU: NVIDIA GeForce RTX 3060
```

### C. Download Dataset

```powershell
# Tạo thư mục
mkdir dataset\images\train
mkdir dataset\images\val
mkdir dataset\labels\train
mkdir dataset\labels\val

# Download từ Google Drive:
# 1. Vào link đã save
# 2. Download file zip
# 3. Giải nén vào dataset/
```

### D. Training

```powershell
# Training với RTX 3060 (12GB VRAM)
python train_license_plate.py
```

**Cấu hình cho RTX 3060:**

```python
# Trong train_license_plate.py
train_model(
    epochs=100,
    batch=16,      # RTX 3060 12GB: dùng 16-24
    model_size='s', # YOLOv8s: cân bằng tốc độ/độ chính xác
    imgsz=640
)
```

**Thời gian ước tính:**

- 1000 ảnh, 100 epochs, batch=16 → ~2-3 giờ

### E. Theo Dõi Training

```powershell
# Training sẽ tự động lưu:
# runs/train/license_plate_vietnam/weights/best.pt
# runs/train/license_plate_vietnam/results.png

# Có thể mở TensorBoard (optional)
tensorboard --logdir runs/train
```

---

## 💾 Bước 4: LƯU MODEL SAU TRAINING

### Cách 1: Google Drive (Dễ nhất)

```powershell
# Upload best.pt lên Drive
# File tại: runs/train/license_plate_vietnam/weights/best.pt
# Share link để download về nhà
```

### Cách 2: Git LFS

```powershell
# Cài Git LFS (nếu chưa có)
git lfs install

# Track file .pt
git lfs track "*.pt"

# Add, commit, push
git add .gitattributes
git add runs/train/*/weights/best.pt
git commit -m "Add trained model"
git push origin master
```

### Cách 3: GitHub Release

```
1. Vào repo trên GitHub
2. Releases → Create new release
3. Tag: v1.0
4. Upload best.pt
5. Publish
```

---

## 🏠 Bước 5: VỀ NHÀ - LẤY MODEL

### Cách 1: Từ Google Drive

```powershell
cd E:\clone\ALPR
# Download best.pt từ Drive
# Đặt vào: models/best.pt
```

### Cách 2: Từ Git

```powershell
cd E:\clone\ALPR
git pull origin master

# Model sẽ tự động pull về (nếu dùng Git LFS)
```

### Test Model

```powershell
# Test với ảnh
python test_single_image.py

# Hoặc test trong code
python
>>> from ultralytics import YOLO
>>> model = YOLO('models/best.pt')
>>> results = model('test.jpg')
>>> results[0].show()
```

---

## 🔄 Workflow Tóm Tắt

```
TẠI NHÀ:
│
├─ 1. Viết/sửa code
│   └─ git add .
│   └─ git commit -m "Update code"
│   └─ git push origin master
│
└─ 2. Upload dataset lên Drive
    └─ Share link

TẠI TIỆM NET:
│
├─ 3. git clone (lần đầu)
│   └─ hoặc git pull (lần sau)
│
├─ 4. Download dataset từ Drive
│
├─ 5. pip install requirements
│
├─ 6. python train_license_plate.py
│   └─ Chờ 2-3 giờ...
│
└─ 7. Upload model lên Drive/Git
    └─ runs/train/*/weights/best.pt

VỀ NHÀ:
│
└─ 8. Download model
    └─ Test model
    └─ Deploy/sử dụng
```

---

## 💡 Tips

### 1. Commit Thường Xuyên

```powershell
# Sau mỗi thay đổi quan trọng
git add .
git commit -m "Fix bug in training script"
git push
```

### 2. Branches (Nâng cao)

```powershell
# Tạo branch cho thử nghiệm
git checkout -b experiment-config
# ... thay đổi code ...
git add .
git commit -m "Try new config"
git push origin experiment-config

# Merge về master sau khi test OK
git checkout master
git merge experiment-config
```

### 3. Xử Lý Conflict

```powershell
# Nếu có conflict khi pull
git pull origin master

# Sửa file conflict thủ công
# Sau đó:
git add .
git commit -m "Resolve conflict"
git push
```

### 4. Backup Code Trước Training

```powershell
# Tại tiệm net, trước khi train
git add .
git commit -m "Before training session $(Get-Date -Format 'yyyy-MM-dd')"
git push
```

---

## ⚠️ Lưu Ý Quan Trọng

1. **Dataset KHÔNG push lên Git:**

   - Quá nặng (GB)
   - Dùng Drive/Mega

2. **Model có thể push nếu nhỏ:**

   - < 100MB: Git LFS OK
   - > 100MB: Dùng Drive

3. **Private Repo:**

   - Đảm bảo repo là Private
   - Tránh lộ code/data

4. **Credentials:**
   - Lưu ý đăng nhập GitHub tại tiệm
   - Dùng Personal Access Token thay password

---

## 🚀 Quick Commands

```powershell
# === TẠI NHÀ ===
git add .
git commit -m "Update training code"
git push origin master

# === TẠI TIỆM NET (lần đầu) ===
git clone https://github.com/YOUR_USERNAME/ALPR-Training.git
cd ALPR-Training
pip install -r requirements_training.txt

# === TẠI TIỆM NET (lần sau) ===
cd ALPR-Training
git pull origin master

# === SAU KHI TRAIN ===
# Upload best.pt lên Drive
# Hoặc:
git add runs/train/*/weights/best.pt
git commit -m "Add trained model"
git push origin master
```

---

**Chúc bạn training thành công với RTX 3060! 🎯**
