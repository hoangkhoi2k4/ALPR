# Import All the Required Libraries
import time
import cv2
import numpy as np
import os
import glob
from pathlib import Path

from detections import VehicleDetection, LicencePlateDetection

def process_image(image_path, licence_plate_detector, output_dir):
    start_time = time.perf_counter()

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Không đọc được ảnh: {image_path}")
        return None

    # Detect plates
    licence_plate_detections, licence_plate_texts = licence_plate_detector.detect_frame(
        frame, [([0, 0, frame.shape[1], frame.shape[0]], 'dummy')]
    )
    
    # Draw results
    frame = licence_plate_detector.draw_bboxes(frame, licence_plate_detections, licence_plate_texts)
    
    # Save output
    image_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{image_name}_output.jpg")
    cv2.imwrite(output_path, frame)
    
    elapsed = time.perf_counter() - start_time
    
    # Extract confidence scores from texts (format: "text (confidence: 0.XX)")
    confidences = []
    clean_texts = []
    for text in licence_plate_texts:
        if "(confidence:" in text:
            parts = text.split("(confidence:")
            clean_text = parts[0].strip()
            conf_str = parts[1].replace(")", "").strip()
            try:
                confidence = float(conf_str)
                confidences.append(confidence)
                clean_texts.append(clean_text)
            except:
                confidences.append(0.0)
                clean_texts.append(text)
        else:
            confidences.append(0.0)
            clean_texts.append(text)
    
    return {
        'time': elapsed,
        'num_plates': len(licence_plate_detections),
        'texts': clean_texts,
        'confidences': confidences,
        'full_texts': licence_plate_texts
    }

def main():
    print("=" * 80)
    print("AUTOMATIC LICENSE PLATE RECOGNITION - BATCH PROCESSING")
    print("=" * 80)
    
    # Setup
    input_folder = "input_images"
    output_folder = "output_images"
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not image_files:
        print(f"Không tìm thấy ảnh nào trong folder '{input_folder}'")
        return
    
    print(f"\nTìm thấy {len(image_files)} ảnh trong folder '{input_folder}'")
    print(f"Output sẽ được lưu vào folder '{output_folder}'")
    
    # Init detector (chỉ khởi tạo 1 lần)
    print(f"\nĐang khởi tạo model...")
    init_start = time.perf_counter()
    licence_plate_detector = LicencePlateDetection(model_path='models/best.pt', ocr_engine='paddle')
    init_time = time.perf_counter() - init_start
    print(f"Model đã sẵn sàng (khởi tạo: {init_time:.2f}s)")
    
    # Process each image
    print(f"\n{'='*80}")
    print(f"{'STT':<5} {'Tên file':<30} {'Thời gian':<12} {'Confidence':<12} {'Kết quả'}")
    print(f"{'='*80}")
    
    results = []
    total_processing_time = 0
    all_confidences = []
    cnt = 0
    for idx, image_path in enumerate(image_files, 1):
        cnt += 1
        if cnt > 20:
            break
        image_name = Path(image_path).name

        result = process_image(image_path, licence_plate_detector, output_folder)

        if result:
            total_processing_time += result['time']
            results.append(result)
            
            # Collect confidences
            all_confidences.extend(result['confidences'])
            
            # Display result
            if result['confidences']:
                avg_conf = sum(result['confidences']) / len(result['confidences'])
                conf_str = f"{avg_conf:.2f}"
            else:
                conf_str = "N/A"
            
            texts_str = ", ".join(result['full_texts']) if result['full_texts'] else "N/A"
            print(f"{idx:<5} {image_name:<30} {result['time']:>8.2f}s    {conf_str:<12} {texts_str}")
        else:
            print(f"{idx:<5} {image_name:<30} {'FAILED':<12}")
    
    # Summary Statistics
    print(f"{'='*80}")
    print(f"\nTHỐNG KÊ TỔNG QUAN:")
    print(f"  • Tổng số ảnh xử lý: {len(results)}/{len(image_files)}")
    print(f"  • Tổng thời gian xử lý: {total_processing_time:.2f}s")
    
    if results:
        avg_time = total_processing_time / len(results)
        print(f"  • Thời gian trung bình: {avg_time:.2f}s/ảnh")
        min_time = min(r['time'] for r in results)
        max_time = max(r['time'] for r in results)
        print(f"  • Nhanh nhất: {min_time:.2f}s")
        print(f"  • Chậm nhất: {max_time:.2f}s")
        
        total_plates = sum(r['num_plates'] for r in results)
        print(f"  • Tổng số biển phát hiện: {total_plates}")
    
    # Confidence Statistics
    if all_confidences:
        # Remove 0.0 confidences (N/A cases)
        valid_confidences = [c for c in all_confidences if c > 0]
        
        if valid_confidences:
            print(f"\n📈 THỐNG KÊ ĐỘ CHÍNH XÁC (OCR Confidence):")
            print(f"  • Tổng số biển có confidence: {len(valid_confidences)}/{len(all_confidences)}")
            
            # Average confidence
            avg_confidence = sum(valid_confidences) / len(valid_confidences)
            print(f"  • Confidence trung bình: {avg_confidence:.4f} ({avg_confidence*100:.2f}%)")
            
            # Min/Max confidence
            min_conf = min(valid_confidences)
            max_conf = max(valid_confidences)
            print(f"  • Thấp nhất: {min_conf:.4f} ({min_conf*100:.2f}%)")
            print(f"  • Cao nhất: {max_conf:.4f} ({max_conf*100:.2f}%)")
            
            # Confidence distribution
            excellent = sum(1 for c in valid_confidences if c >= 0.95)
            good = sum(1 for c in valid_confidences if 0.85 <= c < 0.95)
            fair = sum(1 for c in valid_confidences if 0.70 <= c < 0.85)
            poor = sum(1 for c in valid_confidences if c < 0.70)
            
            print(f"\n  📊 Phân phối Confidence:")
            print(f"    🟢 Xuất sắc (≥0.95):     {excellent}/{len(valid_confidences)} ({excellent/len(valid_confidences)*100:.1f}%)")
            print(f"    🔵 Tốt (0.85-0.95):      {good}/{len(valid_confidences)} ({good/len(valid_confidences)*100:.1f}%)")
            print(f"    🟡 Khá (0.70-0.85):      {fair}/{len(valid_confidences)} ({fair/len(valid_confidences)*100:.1f}%)")
            print(f"    🔴 Yếu (<0.70):          {poor}/{len(valid_confidences)} ({poor/len(valid_confidences)*100:.1f}%)")
            
            # Success rate (confidence >= 0.85)
            success_count = excellent + good
            success_rate = success_count / len(valid_confidences) * 100
            print(f"\n  🎯 Tỉ lệ đọc tốt (≥0.85): {success_count}/{len(valid_confidences)} ({success_rate:.1f}%)")
            
            # List low confidence cases
            low_conf_results = []
            for r in results:
                for i, conf in enumerate(r['confidences']):
                    if conf > 0 and conf < 0.85:
                        low_conf_results.append({
                            'text': r['full_texts'][i],
                            'confidence': conf
                        })
            
            if low_conf_results:
                print(f"\n  ⚠️  Các trường hợp confidence thấp (<0.85):")
                for item in low_conf_results[:10]:  # Show first 10
                    print(f"    • {item['text']} (confidence: {item['confidence']:.2f})")
                
                if len(low_conf_results) > 10:
                    print(f"    ... và {len(low_conf_results) - 10} trường hợp khác")
    
    print(f"\n✅ Hoàn thành! Output đã lưu tại: {output_folder}/")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()