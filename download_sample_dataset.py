"""
Script hỗ trợ download dataset mẫu từ các nguồn công khai
"""
import os
import requests
import zipfile
from pathlib import Path

def download_from_roboflow():
    """
    Hướng dẫn download từ Roboflow
    """
    print("="*60)
    print("DOWNLOAD DATASET TỪ ROBOFLOW")
    print("="*60)
    print("""
1. Truy cập: https://universe.roboflow.com/

2. Tìm kiếm: "vietnam license plate" hoặc "vietnamese license plate"

3. Chọn dataset phù hợp (xem số lượng ảnh, chất lượng)

4. Click "Download Dataset"

5. Chọn format: "YOLOv8" (quan trọng!)

6. Download về máy

7. Giải nén vào thư mục dự án:
   - Train images -> dataset/images/train/
   - Train labels -> dataset/labels/train/
   - Valid images -> dataset/images/val/
   - Valid labels -> dataset/labels/val/
   - Test images -> dataset/images/test/
   - Test labels -> dataset/labels/test/

8. File data.yaml -> copy vào dataset/data.yaml
   (hoặc dùng file có sẵn từ collect_dataset.py)
    """)

def download_from_kaggle():
    """
    Hướng dẫn download từ Kaggle
    """
    print("\n" + "="*60)
    print("DOWNLOAD DATASET TỪ KAGGLE")
    print("="*60)
    print("""
1. Tạo tài khoản Kaggle (miễn phí): https://www.kaggle.com/

2. Vào: https://www.kaggle.com/datasets

3. Tìm: "vietnam license plate dataset"

4. Click dataset -> "Download"

5. Giải nén và chuyển về format YOLO nếu cần:
   - Mỗi ảnh có 1 file .txt tương ứng
   - Format: class x_center y_center width height (normalized 0-1)
   - Ví dụ: 0 0.5 0.5 0.3 0.2

6. Đặt vào thư mục dataset/ theo cấu trúc chuẩn
    """)

def create_sample_annotation():
    """
    Tạo file mẫu cho annotation
    """
    print("\n" + "="*60)
    print("FORMAT ANNOTATION (LABEL)")
    print("="*60)
    
    sample_dir = Path("dataset/annotation_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo file mẫu
    sample_content = """# File label YOLO format
# Tên file: image_001.txt (tương ứng với image_001.jpg)

# Format: class x_center y_center width height
# Tất cả giá trị đã normalized (0-1)

# Ví dụ 1: Biển số ở giữa ảnh
0 0.5 0.5 0.3 0.15

# Ví dụ 2: Biển số góc trên bên trái
0 0.25 0.25 0.2 0.1

# Ví dụ 3: Biển số góc dưới bên phải  
0 0.75 0.75 0.25 0.12

# Giải thích:
# - class = 0 (license_plate - chỉ có 1 class)
# - x_center: tọa độ X tâm biển số (0=trái, 1=phải)
# - y_center: tọa độ Y tâm biển số (0=trên, 1=dưới)
# - width: chiều rộng biển số (tỷ lệ so với ảnh)
# - height: chiều cao biển số (tỷ lệ so với ảnh)
"""
    
    sample_file = sample_dir / "sample_label.txt"
    with open(sample_file, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"✓ Đã tạo file mẫu: {sample_file}")
    print("\n💡 Mở file này để xem format label YOLO")

def list_free_datasets():
    """
    Liệt kê các dataset miễn phí có thể dùng
    """
    print("\n" + "="*60)
    print("CÁC DATASET BIỂN SỐ XE MIỄN PHÍ")
    print("="*60)
    
    datasets = [
        {
            "name": "Vietnamese License Plate Dataset",
            "source": "Roboflow Universe",
            "url": "https://universe.roboflow.com/search?q=vietnam+license+plate",
            "images": "1000-3000",
            "format": "YOLO",
            "note": "Nhiều dataset khác nhau, chọn cái có nhiều ảnh nhất"
        },
        {
            "name": "License Plate Recognition Dataset",
            "source": "Kaggle",
            "url": "https://www.kaggle.com/datasets/search?q=license+plate+vietnam",
            "images": "Varies",
            "format": "Mixed (có thể cần convert)",
            "note": "Tìm dataset có tag 'vietnam' hoặc 'vietnamese'"
        },
        {
            "name": "Open Images Dataset V7",
            "source": "Google",
            "url": "https://storage.googleapis.com/openimages/web/index.html",
            "images": "Large",
            "format": "Need filtering",
            "note": "Dataset lớn, cần filter ra ảnh xe và biển số"
        },
    ]
    
    for i, ds in enumerate(datasets, 1):
        print(f"\n{i}. {ds['name']}")
        print(f"   Source: {ds['source']}")
        print(f"   URL: {ds['url']}")
        print(f"   Images: ~{ds['images']}")
        print(f"   Format: {ds['format']}")
        print(f"   Note: {ds['note']}")

def check_dataset_structure():
    """
    Kiểm tra cấu trúc dataset đã đúng chưa
    """
    print("\n" + "="*60)
    print("KIỂM TRA CẤU TRÚC DATASET")
    print("="*60)
    
    required_dirs = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val",
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            file_count = len(os.listdir(dir_path))
            print(f"✓ {dir_path}: {file_count} files")
            
            if file_count == 0:
                print(f"  ⚠️  Thư mục trống!")
                all_good = False
        else:
            print(f"✗ {dir_path}: Không tồn tại")
            all_good = False
    
    # Kiểm tra data.yaml
    if os.path.exists("dataset/data.yaml"):
        print(f"✓ dataset/data.yaml: Tồn tại")
    else:
        print(f"✗ dataset/data.yaml: Không tồn tại")
        all_good = False
    
    print("\n" + "="*60)
    if all_good:
        print("✅ DATASET ĐÃ SẴN SÀNG CHO TRAINING!")
    else:
        print("⚠️  DATASET CHƯA ĐẦY ĐỦ!")
        print("💡 Chạy: python collect_dataset.py để tạo cấu trúc")
        print("💡 Sau đó download dataset và đặt đúng vị trí")

if __name__ == "__main__":
    print("📥 HƯỚNG DẪN DOWNLOAD DATASET BIỂN SỐ XE\n")
    
    # Liệt kê dataset
    list_free_datasets()
    
    # Hướng dẫn Roboflow
    download_from_roboflow()
    
    # Hướng dẫn Kaggle
    download_from_kaggle()
    
    # Tạo file mẫu
    create_sample_annotation()
    
    # Kiểm tra dataset
    check_dataset_structure()
    
    print("\n" + "="*60)
    print("📌 BƯỚC TIẾP THEO")
    print("="*60)
    print("""
1. Download dataset từ Roboflow hoặc Kaggle
2. Giải nén và đặt vào đúng thư mục
3. Chạy: python download_sample_dataset.py (file này) để kiểm tra
4. Nếu OK, chạy: python train_license_plate.py để training

💡 Dataset càng nhiều, model càng chính xác!
   Khuyến nghị: Ít nhất 1000 ảnh train, 200 ảnh val
    """)
