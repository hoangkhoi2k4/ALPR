"""
Script training model nhận diện biển số xe Việt Nam với CUDA
"""
import torch
from ultralytics import YOLO
import os

def check_gpu():
    """Kiểm tra GPU/CUDA"""
    print("="*60)
    print("KIỂM TRA GPU/CUDA")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        print(f"✅ CUDA version: {torch.version.cuda}")
        print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
        print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"✅ PyTorch version: {torch.__version__}")
        return True
    else:
        print("❌ CUDA không khả dụng!")
        print("💡 Cài đặt:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return False

def train_model(
    data_yaml='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    model_size='n',  # n, s, m, l, x
    device='0',  # 0 = GPU đầu tiên
    project='runs/train',
    name='license_plate_vietnam'
):
    """
    Train YOLOv8 model
    
    Args:
        data_yaml: Đường dẫn file data.yaml
        epochs: Số epoch (100-300 tốt)
        imgsz: Kích thước ảnh training (640 mặc định)
        batch: Batch size (tùy VRAM GPU, 16-32 tốt)
        model_size: Kích thước model (n=nano, s=small, m=medium, l=large, x=xlarge)
        device: GPU device (0, 1, 2,... hoặc 'cpu')
        project: Thư mục lưu kết quả
        name: Tên experiment
    """
    
    print("\n" + "="*60)
    print("BẮT ĐẦU TRAINING MODEL NHẬN DIỆN BIỂN SỐ XE")
    print("="*60)
    
    # Kiểm tra file data.yaml
    if not os.path.exists(data_yaml):
        print(f"❌ Không tìm thấy file: {data_yaml}")
        print("💡 Chạy collect_dataset.py trước để tạo cấu trúc!")
        return
    
    # Kiểm tra GPU
    if not check_gpu() and device != 'cpu':
        print("\n⚠️  Chuyển sang training bằng CPU (sẽ rất chậm)")
        device = 'cpu'
    
    # Load model
    model_name = f'yolov8{model_size}.pt'
    print(f"\n📦 Loading model: {model_name}")
    model = YOLO(model_name)
    
    print(f"\n⚙️  CẤU HÌNH TRAINING:")
    print(f"  • Model: YOLOv8{model_size}")
    print(f"  • Epochs: {epochs}")
    print(f"  • Image size: {imgsz}")
    print(f"  • Batch size: {batch}")
    print(f"  • Device: {device}")
    print(f"  • Dataset: {data_yaml}")
    
    # Training
    print(f"\n🚀 BẮT ĐẦU TRAINING...")
    print("="*60)
    
    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            patience=50,  # Early stopping patience
            save=True,
            save_period=10,  # Lưu checkpoint mỗi 10 epochs
            cache=True,  # Cache ảnh để training nhanh hơn
            pretrained=True,
            optimizer='auto',
            verbose=True,
            seed=42,
            deterministic=True,
            workers=8,
            # Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
        )
        
        print("\n" + "="*60)
        print("✅ TRAINING HOÀN TẤT!")
        print("="*60)
        print(f"\n📊 Kết quả:")
        print(f"  • Best model: {project}/{name}/weights/best.pt")
        print(f"  • Last model: {project}/{name}/weights/last.pt")
        print(f"  • Metrics: {project}/{name}/results.png")
        
        return results
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        print("\n💡 Một số lỗi thường gặp:")
        print("  • Thiếu dataset: Đảm bảo có ảnh trong dataset/images/train")
        print("  • Thiếu labels: Đảm bảo có file .txt trong dataset/labels/train")
        print("  • Hết VRAM: Giảm batch size (16 -> 8 -> 4)")
        print("  • File data.yaml sai: Kiểm tra đường dẫn và format")

def validate_model(model_path, data_yaml='dataset/data.yaml'):
    """Validate model trên test set"""
    print("\n" + "="*60)
    print("VALIDATE MODEL")
    print("="*60)
    
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    
    print(f"\n📊 Kết quả validation:")
    print(f"  • mAP50: {results.box.map50:.4f}")
    print(f"  • mAP50-95: {results.box.map:.4f}")
    print(f"  • Precision: {results.box.mp:.4f}")
    print(f"  • Recall: {results.box.mr:.4f}")

if __name__ == "__main__":
    # Kiểm tra GPU trước
    check_gpu()
    
    print("\n" + "="*60)
    print("HƯỚNG DẪN SỬ DỤNG")
    print("="*60)
    print("""
1. Chuẩn bị dataset:
   python collect_dataset.py
   python split_dataset.py

2. Training (tại tiệm net có GPU):
   python train_license_plate.py
   
3. Tùy chỉnh training:
   • Model nhỏ, nhanh: model_size='n' hoặc 's'
   • Model chính xác: model_size='m' hoặc 'l'
   • Ít VRAM: giảm batch=8 hoặc 4
   • Nhiều VRAM: tăng batch=32 hoặc 64
   
4. Training thử nghiệm nhanh (10 epochs):
    """)
    
    # Uncomment để chạy training
    # train_model(
    #     epochs=10,  # Test thử 10 epochs
    #     batch=16,
    #     model_size='n',  # Model nhỏ nhất
    #     imgsz=640
    # )
    
    print("\n💡 Uncomment dòng train_model() ở cuối file để bắt đầu training!")
    print("💡 Hoặc import và gọi: train_model(epochs=100, batch=16)")
