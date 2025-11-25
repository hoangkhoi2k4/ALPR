"""
Script tự động hóa Git workflow cho training AI
Dùng để sync giữa máy nhà và tiệm net
"""
import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, show_output=True):
    """Chạy command và trả về kết quả"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if show_output and result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return False

def check_git():
    """Kiểm tra Git đã cài chưa"""
    result = subprocess.run("git --version", shell=True, capture_output=True)
    if result.returncode == 0:
        print(f"✓ {result.stdout.decode().strip()}")
        return True
    else:
        print("❌ Git chưa được cài đặt!")
        print("💡 Download tại: https://git-scm.com/download/win")
        return False

def init_gitignore():
    """Tạo .gitignore để loại bỏ file không cần thiết"""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# Jupyter Notebook
.ipynb_checkpoints

# Dataset (không push dataset lên git - quá nặng)
dataset/images/
dataset/labels/
dataset/raw_images/
dataset/raw_labels/
*.jpg
*.jpeg
*.png
*.bmp

# Chỉ giữ cấu trúc thư mục
!dataset/images/train/.gitkeep
!dataset/images/val/.gitkeep
!dataset/images/test/.gitkeep

# Training results
runs/
*.pt  # Model files - quá nặng, dùng Git LFS hoặc upload riêng
*.pth
*.onnx
*.engine

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
*.tmp
*.temp
"""
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        print("⚠️  .gitignore đã tồn tại")
        response = input("Ghi đè? (y/n): ")
        if response.lower() != 'y':
            return
    
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write(gitignore_content)
    
    print("✓ Đã tạo .gitignore")
    
    # Tạo file .gitkeep cho thư mục rỗng
    dirs_to_keep = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/test"
    ]
    
    for dir_path in dirs_to_keep:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        gitkeep = Path(dir_path) / ".gitkeep"
        gitkeep.touch()
    
    print("✓ Đã tạo .gitkeep cho các thư mục")

def setup_git_lfs():
    """Setup Git LFS cho file lớn (model)"""
    print("\n" + "="*60)
    print("GIT LFS - Cho file lớn (model .pt)")
    print("="*60)
    
    # Check Git LFS
    result = subprocess.run("git lfs version", shell=True, capture_output=True)
    
    if result.returncode != 0:
        print("❌ Git LFS chưa cài")
        print("💡 Cài Git LFS:")
        print("   1. Download: https://git-lfs.github.com/")
        print("   2. Hoặc: winget install GitHub.GitLFS")
        print("   3. Sau đó chạy: git lfs install")
        return False
    
    print("✓ Git LFS đã cài")
    
    # Setup LFS tracking
    print("\n📦 Setup tracking file .pt (model)")
    run_command("git lfs install")
    run_command("git lfs track '*.pt'")
    run_command("git lfs track '*.pth'")
    
    print("\n✓ Đã setup Git LFS")
    print("💡 File .pt sẽ được lưu riêng, không làm repo nặng")
    
    return True

def push_to_github():
    """Push code lên GitHub"""
    print("\n" + "="*60)
    print("PUSH CODE LÊN GITHUB")
    print("="*60)
    
    # Check remote
    result = subprocess.run("git remote -v", shell=True, capture_output=True, text=True)
    
    if not result.stdout.strip():
        print("⚠️  Chưa có remote repository")
        print("\n💡 Hướng dẫn:")
        print("1. Tạo repo mới trên GitHub: https://github.com/new")
        print("2. Đặt tên: ALPR-Training (hoặc tên khác)")
        print("3. Chọn Private (để bảo mật)")
        print("4. Chạy lệnh:")
        print("   git remote add origin https://github.com/YOUR_USERNAME/ALPR-Training.git")
        return False
    
    print("Remote repository:")
    print(result.stdout)
    
    # Git add, commit, push
    print("\n📤 Đang push code...")
    
    if run_command("git add ."):
        print("✓ Git add")
    
    commit_msg = input("\nCommit message (Enter = 'Update training code'): ").strip()
    if not commit_msg:
        commit_msg = "Update training code"
    
    if run_command(f'git commit -m "{commit_msg}"'):
        print("✓ Git commit")
    else:
        print("⚠️  Không có thay đổi hoặc lỗi commit")
    
    branch = "master"  # hoặc main
    if run_command(f"git push -u origin {branch}"):
        print(f"✓ Git push lên {branch}")
        return True
    else:
        print("❌ Push thất bại")
        print("💡 Có thể cần: git pull origin master --rebase")
        return False

def pull_from_github():
    """Pull code mới từ GitHub"""
    print("\n" + "="*60)
    print("PULL CODE TỪ GITHUB")
    print("="*60)
    
    branch = "master"  # hoặc main
    
    print(f"📥 Đang pull từ {branch}...")
    if run_command(f"git pull origin {branch}"):
        print("✓ Pull thành công")
        return True
    else:
        print("❌ Pull thất bại")
        return False

def clone_repo():
    """Clone repo về máy mới (tại tiệm net)"""
    print("\n" + "="*60)
    print("CLONE REPO VỀ MÁY MỚI (TẠI TIỆM NET)")
    print("="*60)
    
    repo_url = input("Nhập GitHub repo URL: ").strip()
    
    if not repo_url:
        print("❌ Chưa nhập URL")
        return False
    
    print(f"\n📥 Đang clone {repo_url}...")
    if run_command(f"git clone {repo_url}"):
        print("✓ Clone thành công")
        print("\n💡 Bước tiếp theo tại tiệm net:")
        print("1. cd vào thư mục vừa clone")
        print("2. pip install -r requirements_training.txt")
        print("3. Download dataset riêng (từ Drive/Mega)")
        print("4. python train_license_plate.py")
        return True
    else:
        print("❌ Clone thất bại")
        return False

def save_model_to_cloud():
    """Hướng dẫn upload model lên cloud"""
    print("\n" + "="*60)
    print("LƯU MODEL SAU KHI TRAIN")
    print("="*60)
    
    print("""
Model .pt thường nặng (5-50MB), có thể push lên Git với LFS hoặc dùng:

1. GOOGLE DRIVE (Khuyến nghị):
   • Upload file best.pt lên Drive
   • Share link, copy về máy nhà
   • Nhanh, đơn giản

2. GIT LFS (Nếu đã setup):
   • git add runs/train/*/weights/best.pt
   • git commit -m "Add trained model"
   • git push
   • Tại máy nhà: git pull

3. MEGA.NZ / DROPBOX:
   • Upload model
   • Download tại máy khác

4. GITHUB RELEASE:
   • Vào repo trên GitHub
   • Releases -> Create new release
   • Attach file best.pt
   • Download ở máy khác

💡 Khuyến nghị: Google Drive (dễ nhất, không cần setup)
    """)

def show_workflow():
    """Hiển thị workflow đầy đủ"""
    print("\n" + "="*60)
    print("WORKFLOW GIT CHO TRAINING AI")
    print("="*60)
    
    print("""
📍 TẠI NHÀ (Chuẩn bị):
─────────────────────────────
1. Chuẩn bị code và dataset
   python collect_dataset.py
   python split_dataset.py

2. Push code lên GitHub
   git add .
   git commit -m "Prepare training"
   git push origin master

3. Upload dataset lên Google Drive
   (Dataset quá nặng, không push lên Git)

📍 TẠI TIỆM NET (Training):
─────────────────────────────
1. Clone repo
   git clone https://github.com/YOUR_USERNAME/ALPR-Training.git
   cd ALPR-Training

2. Cài môi trường
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   pip install -r requirements_training.txt

3. Download dataset từ Drive về
   (Đặt vào dataset/images/ và dataset/labels/)

4. Training
   python train_license_plate.py

5. Lưu model
   • Upload best.pt lên Google Drive
   • Hoặc: git add runs/train/*/weights/best.pt
            git commit -m "Add trained model"
            git push

📍 VỀ NHÀ (Lấy model):
─────────────────────────────
1. Download model từ Drive
   Hoặc: git pull

2. Test model
   python test_single_image.py

💡 LƯU Ý:
• Dataset để trên Drive, không push Git (quá nặng)
• Code thì push Git (dễ sync)
• Model có thể Git LFS hoặc Drive
• Nhớ commit thường xuyên để backup
    """)

def main():
    """Menu chính"""
    print("🔧 GIT WORKFLOW HELPER - TRAINING AI")
    print("="*60)
    
    if not check_git():
        return
    
    while True:
        print("\n" + "="*60)
        print("CHỌN THAO TÁC:")
        print("="*60)
        print("1. Setup Git (tạo .gitignore, Git LFS)")
        print("2. Push code lên GitHub")
        print("3. Pull code từ GitHub")
        print("4. Clone repo (tại tiệm net)")
        print("5. Xem workflow đầy đủ")
        print("6. Hướng dẫn lưu model")
        print("0. Thoát")
        
        choice = input("\nChọn (0-6): ").strip()
        
        if choice == '1':
            init_gitignore()
            setup_git_lfs()
        elif choice == '2':
            push_to_github()
        elif choice == '3':
            pull_from_github()
        elif choice == '4':
            clone_repo()
        elif choice == '5':
            show_workflow()
        elif choice == '6':
            save_model_to_cloud()
        elif choice == '0':
            print("👋 Bye!")
            break
        else:
            print("❌ Lựa chọn không hợp lệ")

if __name__ == "__main__":
    main()
