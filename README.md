# 🚗 Automatic License Plate Recognition (ALPR)

Hệ thống nhận diện biển số xe tự động cho biển số Việt Nam sử dụng YOLOv8 và PaddleOCR.

## ✨ Tính năng

- ✅ Phát hiện biển số xe với YOLOv8
- ✅ Nhận dạng ký tự biển số với PaddleOCR (hỗ trợ tiếng Việt)
- ✅ Xử lý ảnh độ phân giải thấp với ensemble preprocessing
- ✅ Auto-correction cho lỗi OCR phổ biến (6/9, 2/Z, O/0, I/1)
- ✅ Batch processing với thống kê hiệu năng
- ✅ Hỗ trợ format biển số Việt Nam: `XX-YZ NNNNN.NN`

## 📊 Hiệu suất

- **Độ chính xác**: ~90% trên ảnh low-resolution
- **Tốc độ xử lý**: ~2.8s/ảnh (trung bình)
- **OCR Confidence**: 95%+ trên hầu hết ảnh

## 🛠️ Cài đặt

### Yêu cầu

- Python 3.8+
- CUDA (optional, để tăng tốc)

### Các bước cài đặt

1. Clone repository:

```bash
git clone https://github.com/your-username/automatic-plate-recognition.git
cd automatic-plate-recognition
```

2. Tạo virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# hoặc
source .venv/bin/activate  # Linux/Mac
```

3. Cài đặt dependencies:

```bash
pip install -r requirement.txt
```

4. Tải model YOLOv8:
   - Đặt model trained vào folder `models/best.pt`
   - Hoặc train model của bạn với dataset biển số xe

## 🚀 Sử dụng

### Batch Processing

Xử lý tất cả ảnh trong folder `input_images/`:

```bash
python main.py
```

Kết quả:

- Ảnh output: `output_images/`
- Thống kê: confidence scores, thời gian xử lý, độ chính xác

### Output mẫu

```
================================================================================
STT   Tên file                       Thời gian    Confidence   Kết quả
================================================================================
1     carlong_0001.png                   2.85s    1.00         51-G 100.96
2     carlong_0002.png                   2.71s    1.00         51-G 100.96
3     carlong_0003.png                   2.93s    1.00         51-A 654.74
...

📊 THỐNG KÊ TỔNG QUAN:
  • Tổng số ảnh xử lý: 104/104
  • Thời gian trung bình: 2.87s/ảnh
  • Confidence trung bình: 0.9523 (95.23%)
```

## 📁 Cấu trúc thư mục

```
.
├── detections/
│   ├── __init__.py
│   ├── car_detection.py          # YOLOv8 vehicle detection
│   └── licence_plate_detection.py # License plate detection & OCR
├── utils/
│   ├── __init__.py
│   └── video_ultis.py            # Video processing utilities
├── models/
│   └── best.pt                    # YOLOv8 trained model
├── input_images/                  # Input images
├── output_images/                 # Output images with annotations
├── main.py                        # Main batch processing script
├── requirement.txt                # Python dependencies
├── .gitignore
└── README.md
```

## 🔧 Cấu hình

### Ensemble Preprocessing

Code sử dụng 3 phương pháp preprocessing:

1. **Method 1 (Low-res)**: 7x upscaling, CLAHE, sharpening, dual thresholding
2. **Method 2 (Aggressive)**: 9x LANCZOS4, LAB color space, morphology
3. **Method 3 (Super-resolution)**: 9x upscaling, edge detection, aggressive sharpening

### OCR Auto-correction

- **Character mapping**: O→0, I→1, G→6, Z→2, S→5, B→8
- **Format validation**: XX-YZ NNNNN.NN pattern
- **Province code correction**: 2 digits (10-99)
- **Decimal fixing**: removes extra leading digits

## 📝 Format biển số hỗ trợ

- `51-G 100.96` - Format chuẩn
- `68A-028.66` - Biển số mới
- `29-Z1 288.88` - Các tỉnh thành khác
- `86-B1 374.49` - Nhiều format khác

## 🐛 Xử lý lỗi

### Lỗi phổ biến

1. **OCR confidence thấp**:

   - Kiểm tra độ phân giải ảnh input
   - ROI quá nhỏ (<30px height)

2. **Nhầm ký tự**:

   - 6 ↔ 9, 2 ↔ Z, O ↔ 0, I ↔ 1
   - → Auto-correction đã xử lý

3. **YOLO detect sai**:
   - Điều chỉnh confidence threshold
   - Retrain model với data tốt hơn

## 🤝 Đóng góp

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - xem file [LICENSE](LICENSE) để biết thêm chi tiết.

## 👨‍💻 Tác giả

- GitHub: [@your-username](https://github.com/your-username)

## 🙏 Credits

- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - OCR engine
- [OpenCV](https://opencv.org/) - Image processing

---

⭐ Nếu project hữu ích, hãy cho một star nhé!
