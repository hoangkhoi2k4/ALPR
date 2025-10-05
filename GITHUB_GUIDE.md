# 📤 Hướng dẫn đẩy code lên GitHub

## Bước 1: Khởi tạo Git repository (nếu chưa có)

```bash
git init
```

## Bước 2: Thêm remote repository

```bash
# Tạo repository trên GitHub trước, sau đó:
git remote add origin https://github.com/YOUR_USERNAME/automatic-plate-recognition.git
```

## Bước 3: Kiểm tra files sẽ commit

```bash
git status
```

**Files nên commit:**

- ✅ `main.py`
- ✅ `detections/` (tất cả file .py)
- ✅ `utils/` (tất cả file .py)
- ✅ `requirement.txt`
- ✅ `README.md`
- ✅ `LICENSE`
- ✅ `.gitignore`
- ✅ `clean.py`

**Files KHÔNG nên commit:**

- ❌ `.venv/` (virtual environment)
- ❌ `__pycache__/`
- ❌ `debug_plates/`
- ❌ `output_images/`
- ❌ `models/*.pt` (model files quá lớn)
- ❌ `test*.py`, `check*.py`
- ❌ `input_images/` (optional - có thể push một vài ảnh mẫu)

## Bước 4: Commit code

```bash
# Add tất cả files
git add .

# Hoặc add từng file cụ thể
git add main.py detections/ utils/ requirement.txt README.md LICENSE .gitignore

# Commit
git commit -m "Initial commit: Automatic License Plate Recognition system"
```

## Bước 5: Push lên GitHub

```bash
# Push lần đầu
git push -u origin main

# Hoặc nếu branch là master
git push -u origin master
```

## Bước 6: Cập nhật sau này

```bash
# Kiểm tra thay đổi
git status

# Add files đã thay đổi
git add .

# Commit với message mô tả
git commit -m "Fix: Improved OCR accuracy for small license plates"

# Push
git push
```

## 📝 Lưu ý quan trọng

### 1. Model files

Model YOLOv8 (`.pt` files) rất lớn (>100MB) nên:

- ❌ **KHÔNG** push trực tiếp lên GitHub
- ✅ Sử dụng Git LFS (Large File Storage) hoặc
- ✅ Host model trên Google Drive/Dropbox và thêm link download vào README

### 2. Sensitive data

- ❌ KHÔNG commit API keys, passwords
- ❌ KHÔNG commit ảnh cá nhân/nhạy cảm
- ✅ Kiểm tra `.gitignore` trước khi commit

### 3. Clean code

Trước khi commit, chạy:

```bash
python clean.py
```

## 🔗 Tham khảo thêm

- [Git LFS](https://git-lfs.github.com/) - Để push file lớn
- [GitHub Pages](https://pages.github.com/) - Để host demo
- [GitHub Actions](https://github.com/features/actions) - Để tự động test

---

✅ **Checklist trước khi push:**

- [ ] Đã chạy `python clean.py`
- [ ] Đã cập nhật README.md với thông tin của bạn
- [ ] Đã xóa test files và debug folders
- [ ] Đã kiểm tra `.gitignore`
- [ ] Đã test code chạy được
- [ ] Commit message rõ ràng
