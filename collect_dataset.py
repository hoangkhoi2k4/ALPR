"""
Script thu thập và chuẩn bị dataset biển số xe Việt Nam
"""
import os
import shutil
from pathlib import Path

def create_dataset_structure():
    """Tạo cấu trúc thư mục cho dataset"""
    base_dir = Path("dataset")
    
    # Tạo các thư mục cần thiết
    folders = [
        "images/train",
        "images/val", 
        "images/test",
        "labels/train",
        "labels/val",
        "labels/test",
        "raw_images"  # Thư mục chứa ảnh gốc chưa label
    ]
    
    for folder in folders:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)
    
    print("✓ Đã tạo cấu trúc thư mục dataset")
    print("\nCấu trúc:")
    print("dataset/")
    print("  ├── images/")
    print("  │   ├── train/")
    print("  │   ├── val/")
    print("  │   └── test/")
    print("  ├── labels/")
    print("  │   ├── train/")
    print("  │   ├── val/")
    print("  │   └── test/")
    print("  └── raw_images/  <- Đặt ảnh biển số xe vào đây")
    
    return base_dir

def create_data_yaml(base_dir):
    """Tạo file config cho training"""
    yaml_content = """# Dataset config cho biển số xe Việt Nam
path: ./dataset  # dataset root dir
train: images/train  # train images
val: images/val  # val images
test: images/test  # test images (optional)

# Classes
nc: 1  # number of classes
names: ['license_plate']  # class names
"""
    
    yaml_path = base_dir / "data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    
    print(f"\n✓ Đã tạo file config: {yaml_path}")

def download_sample_dataset():
    """Hướng dẫn download dataset mẫu"""
    print("\n" + "="*60)
    print("CÁCH THU THẬP DATASET BIỂN SỐ XE VIỆT NAM")
    print("="*60)
    
    print("\n1. DATASET CÔNG KHAI:")
    print("   • Roboflow: https://universe.roboflow.com/")
    print("     - Tìm 'vietnam license plate' hoặc 'vietnamese license plate'")
    print("     - Download format YOLO")
    
    print("\n   • Kaggle: https://www.kaggle.com/datasets")
    print("     - Tìm 'vietnam license plate dataset'")
    
    print("\n2. TỰ THU THẬP:")
    print("   • Chụp ảnh xe trên đường (đảm bảo tuân thủ quyền riêng tư)")
    print("   • Tìm ảnh xe từ các nguồn hợp pháp")
    print("   • Đặt ảnh vào: dataset/raw_images/")
    
    print("\n3. CÔNG CỤ LABEL:")
    print("   • LabelImg: https://github.com/HumanSignal/labelImg")
    print("   • Roboflow (online): https://roboflow.com/")
    print("   • CVAT: https://www.cvat.ai/")
    
    print("\n4. YÊU CẦU TỐI THIỂU:")
    print("   • Train: ~1000-2000 ảnh")
    print("   • Val: ~200-300 ảnh")
    print("   • Test: ~100-200 ảnh")
    
    print("\n" + "="*60)

def split_dataset():
    """Hướng dẫn chia dataset"""
    print("\n" + "="*60)
    print("SAU KHI CÓ ẢNH VÀ LABELS")
    print("="*60)
    print("\nChạy script này để tự động chia dataset:")
    print("  python split_dataset.py")
    print("\nHoặc thủ công:")
    print("  - 70% ảnh -> images/train + labels/train")
    print("  - 20% ảnh -> images/val + labels/val")
    print("  - 10% ảnh -> images/test + labels/test")

if __name__ == "__main__":
    print("🚗 CHUẨN BỊ DATASET BIỂN SỐ XE VIỆT NAM\n")
    
    # Tạo cấu trúc thư mục
    base_dir = create_dataset_structure()
    
    # Tạo file config
    create_data_yaml(base_dir)
    
    # Hướng dẫn thu thập
    download_sample_dataset()
    
    # Hướng dẫn chia dataset
    split_dataset()
    
    print("\n✅ HOÀN TẤT! Bắt đầu thu thập ảnh biển số xe nhé!")
