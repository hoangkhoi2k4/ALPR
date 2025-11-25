"""
Script tự động chia dataset thành train/val/test
"""
import os
import shutil
import random
from pathlib import Path

def split_dataset(source_images_dir, source_labels_dir, train_ratio=0.7, val_ratio=0.2):
    """
    Chia dataset thành train/val/test
    
    Args:
        source_images_dir: Thư mục chứa ảnh đã label
        source_labels_dir: Thư mục chứa file label tương ứng
        train_ratio: Tỷ lệ train (default: 0.7 = 70%)
        val_ratio: Tỷ lệ validation (default: 0.2 = 20%)
    """
    
    # Đường dẫn
    images_dir = Path(source_images_dir)
    labels_dir = Path(source_labels_dir)
    
    # Thư mục đích
    train_img_dir = Path("dataset/images/train")
    val_img_dir = Path("dataset/images/val")
    test_img_dir = Path("dataset/images/test")
    
    train_lbl_dir = Path("dataset/labels/train")
    val_lbl_dir = Path("dataset/labels/val")
    test_lbl_dir = Path("dataset/labels/test")
    
    # Lấy danh sách file ảnh
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
    
    if not image_files:
        print(f"❌ Không tìm thấy ảnh trong {images_dir}")
        return
    
    print(f"📊 Tìm thấy {len(image_files)} ảnh")
    
    # Shuffle random
    random.shuffle(image_files)
    
    # Tính số lượng
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]
    
    print(f"\n📂 Chia dataset:")
    print(f"  • Train: {len(train_files)} ảnh ({train_ratio*100:.0f}%)")
    print(f"  • Val:   {len(val_files)} ảnh ({val_ratio*100:.0f}%)")
    print(f"  • Test:  {len(test_files)} ảnh ({(1-train_ratio-val_ratio)*100:.0f}%)")
    
    # Copy files
    def copy_files(file_list, dest_img_dir, dest_lbl_dir):
        copied = 0
        for img_path in file_list:
            # Copy ảnh
            shutil.copy2(img_path, dest_img_dir / img_path.name)
            
            # Copy label (nếu có)
            label_name = img_path.stem + '.txt'
            label_path = labels_dir / label_name
            
            if label_path.exists():
                shutil.copy2(label_path, dest_lbl_dir / label_name)
                copied += 1
            else:
                print(f"⚠️  Thiếu label cho: {img_path.name}")
        
        return copied
    
    print("\n🔄 Đang copy files...")
    train_copied = copy_files(train_files, train_img_dir, train_lbl_dir)
    val_copied = copy_files(val_files, val_img_dir, val_lbl_dir)
    test_copied = copy_files(test_files, test_img_dir, test_lbl_dir)
    
    print(f"\n✅ HOÀN TẤT!")
    print(f"  • Train: {train_copied}/{len(train_files)} có label")
    print(f"  • Val:   {val_copied}/{len(val_files)} có label")
    print(f"  • Test:  {test_copied}/{len(test_files)} có label")
    
    if train_copied < len(train_files):
        print(f"\n⚠️  Một số ảnh thiếu label. Cần label đủ trước khi train!")

if __name__ == "__main__":
    print("🔀 CHIA DATASET TỰ ĐỘNG\n")
    
    # Cấu hình
    SOURCE_IMAGES = "dataset/raw_images"  # Thư mục chứa ảnh gốc
    SOURCE_LABELS = "dataset/raw_labels"  # Thư mục chứa label tương ứng
    
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(SOURCE_IMAGES):
        print(f"❌ Không tìm thấy thư mục: {SOURCE_IMAGES}")
        print("💡 Đặt ảnh đã label vào thư mục này trước!")
        
        # Tạo thư mục nếu chưa có
        os.makedirs(SOURCE_IMAGES, exist_ok=True)
        os.makedirs(SOURCE_LABELS, exist_ok=True)
        print(f"✓ Đã tạo thư mục: {SOURCE_IMAGES} và {SOURCE_LABELS}")
        print("\nHướng dẫn:")
        print("  1. Đặt ảnh vào: dataset/raw_images/")
        print("  2. Đặt label (.txt) vào: dataset/raw_labels/")
        print("  3. Chạy lại script này")
    else:
        # Chia dataset
        split_dataset(SOURCE_IMAGES, SOURCE_LABELS)
