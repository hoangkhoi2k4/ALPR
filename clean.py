"""
Script để dọn dẹp các folder debug/test trước khi commit
"""
import os
import shutil

def clean_debug_folders():
    """Xóa các folder debug và test files"""
    folders_to_clean = [
        'debug_plates',
        'output_images',
        'output_videos',
        'output_images_test',
        '__pycache__',
        'tracker_stubs'
    ]
    
    files_to_clean = [
        'results.txt'
    ]
    
    print("🧹 Cleaning debug folders and files...")
    
    # Clean folders
    for folder in folders_to_clean:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"  ✅ Removed: {folder}/")
            except Exception as e:
                print(f"  ❌ Error removing {folder}: {e}")
    
    # Clean files
    for file in files_to_clean:
        if os.path.exists(file):
            try:
                os.remove(file)
                print(f"  ✅ Removed: {file}")
            except Exception as e:
                print(f"  ❌ Error removing {file}: {e}")
    
    # Clean test files
    import glob
    test_files = glob.glob("test*.py") + glob.glob("check*.py")
    for file in test_files:
        try:
            os.remove(file)
            print(f"  ✅ Removed: {file}")
        except Exception as e:
            print(f"  ❌ Error removing {file}: {e}")
    
    print("\n✨ Cleanup complete!")
    print("\n💡 Remember to:")
    print("  1. Keep your trained model (models/best.pt) but add it to .gitignore")
    print("  2. Review .gitignore before committing")
    print("  3. Update README.md with your GitHub username")

if __name__ == "__main__":
    clean_debug_folders()
