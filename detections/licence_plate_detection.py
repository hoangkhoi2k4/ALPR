# Import All the Required Libraries
import cv2
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np

class LicencePlateDetection:
    def __init__(self, model_path, ocr_engine='paddle'):
        self.model = YOLO(model_path)
        if ocr_engine == 'paddle':
            self.ocr = PaddleOCR(use_angle_cls=True, lang='vi') 
            self.ocr_type = 'paddle'
        else:
            self.ocr = easyocr.Reader(['vi'])
            self.ocr_type = 'easy'

    def detect_frame(self, frame, vehicle_detections):
        licence_plate_list = []
        licence_plate_texts = []
        if not vehicle_detections:
            vehicle_detections = [([0, 0, frame.shape[1], frame.shape[0]], 'dummy')]
        
        for veh_bbox, _ in vehicle_detections:
            vx1, vy1, vx2, vy2 = map(int, veh_bbox)
            roi = frame[vy1:vy2, vx1:vx2]
            results = self.model.track(roi, persist=True, conf=0.2)
            bbox_list, text_list = self.process_frame(roi, results[0])
            for i in range(len(bbox_list)):
                bbox = bbox_list[i]
                bbox[0] += vx1
                bbox[1] += vy1
                bbox[2] += vx1
                bbox[3] += vy1
            licence_plate_list.extend(bbox_list)
            licence_plate_texts.extend(text_list)
        return licence_plate_list, licence_plate_texts

    def process_frame(self, frame, results):
        id_name_dict = results.names
        licence_plate_list = []
        licence_plate_texts = []
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]
            if cls_name == "License_Plate":
                licence_plate_list.append(result)
                combined_text = self.run_ocr(frame, result)
                licence_plate_texts.append(combined_text)
        return licence_plate_list, licence_plate_texts

    def run_ocr(self, frame, result):
        x1, y1, x2, y2 = map(int, result)
        cropped_plate = frame[y1:y2, x1:x2]
        
        # Save original crop for debugging
        import os
        debug_dir = "debug_plates"
        os.makedirs(debug_dir, exist_ok=True)
        import time
        timestamp = int(time.time() * 1000)
        cv2.imwrite(f"{debug_dir}/original_{timestamp}.jpg", cropped_plate)
        
        # Ensemble approach: try multiple preprocessing strategies
        candidates = []
        
        # Strategy 1: Low-res optimized (always run first)
        preprocessed1 = self.preprocess_for_lowres(cropped_plate)
        cv2.imwrite(f"{debug_dir}/method1_{timestamp}.jpg", preprocessed1)
        ocr_result1 = self.ocr.ocr(preprocessed1)
        text1, conf1 = self.process_ocr_result(ocr_result1)
        text1_cleaned = self.fix_common_ocr_errors(text1)
        text1_final = self.final_format_correction(text1_cleaned)
        valid1 = self.validate_plate_format(text1_final)
        candidates.append((text1_final, conf1, valid1, "method1"))
        
        # Early exit if method1 gives good result
        if valid1 and conf1 > 0.90:
            print(f"Licence Text: {text1_final} (confidence: {conf1:.2f}, valid: {valid1}, method: method1)")
            return f"{text1_final} (confidence: {conf1:.2f})"  # ← FIX: Return with confidence
        
        # Strategy 2: Aggressive enhancement (run if method1 not good enough)
        preprocessed2 = self.preprocess_aggressive(cropped_plate)
        cv2.imwrite(f"{debug_dir}/method2_{timestamp}.jpg", preprocessed2)
        ocr_result2 = self.ocr.ocr(preprocessed2)
        text2, conf2 = self.process_ocr_result(ocr_result2)
        text2_cleaned = self.fix_common_ocr_errors(text2)
        text2_final = self.final_format_correction(text2_cleaned)
        valid2 = self.validate_plate_format(text2_final)
        candidates.append((text2_final, conf2, valid2, "method2"))
        
        # Early exit if method2 gives good result
        if valid2 and conf2 > 0.90:
            print(f"Licence Text: {text2_final} (confidence: {conf2:.2f}, valid: {valid2}, method: method2)")
            return f"{text2_final} (confidence: {conf2:.2f})"  # ← FIX: Return with confidence
        
        # Strategy 3: Super-resolution (only if both method1 and method2 failed)
        if (not valid1 or conf1 < 0.85) and (not valid2 or conf2 < 0.85):
            print(f"  ⚠️ Low confidence, trying super-resolution...")
            preprocessed3 = self.preprocess_super_resolution(cropped_plate)
            cv2.imwrite(f"{debug_dir}/method3_{timestamp}.jpg", preprocessed3)
            ocr_result3 = self.ocr.ocr(preprocessed3)
            text3, conf3 = self.process_ocr_result(ocr_result3)
            text3_cleaned = self.fix_common_ocr_errors(text3)
            text3_final = self.final_format_correction(text3_cleaned)
            valid3 = self.validate_plate_format(text3_final)
            candidates.append((text3_final, conf3, valid3, "method3"))
        
        # Select best candidate based on validation and confidence
        best_candidate = self.select_best_candidate(candidates)
        result_text, final_confidence, is_valid, method = best_candidate
        
        print(f"Licence Text: {result_text} (confidence: {final_confidence:.2f}, valid: {is_valid}, method: {method})")
        return f"{result_text} (confidence: {final_confidence:.2f})"  # ← FIX: Return with confidence

    
    def final_format_correction(self, text):
        """Final pass to ensure correct Vietnamese plate format"""
        import re
        
        if text == "N/A" or not text:
            return text
        
        # Clean up text
        text_clean = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common punctuation in numbers (but keep original decimal digits)
        text_clean = re.sub(r':(\d)', r'.\1', text_clean)  # 869:88 -> 869.88
        
        # Remove all spaces and dashes for parsing
        text_no_sep = re.sub(r'[\s\-_:·]', '', text_clean)
        
        # Try to match pattern: digits-letters-digits
        # Allow flexible formats: 51G100.96, 511F869.88, 51A654.74, 51G6495.24
        match = re.match(r'^(\d{1,3})([A-Z]{1,2})(\d{3,7}\.?\d{0,2})$', text_no_sep)
        
        if match:
            province = match.group(1)
            letters = match.group(2)
            numbers = match.group(3)
            
            # Fix province code (must be 2 digits, typically 10-99)
            if len(province) == 1:
                province = '5' + province
            elif len(province) == 3:
                province = province[0:2]
            
            try:
                prov_num = int(province)
                if prov_num < 10:
                    province = '0' + province if len(province) == 1 else province
                elif prov_num > 99:
                    province = str(prov_num)[-2:]
            except:
                province = '51'
            
            # Fix numbers
            numbers_clean = numbers
            
            if '.' in numbers:
                parts = numbers.split('.')
                integer_part = parts[0]
                decimal_part = parts[1] if len(parts) > 1 else ''
                
                # Integer part validation: 3-4 digits is standard for Vietnamese plates
                if len(integer_part) == 4:
                    # Check if first digit makes it look wrong
                    first_digit = int(integer_part[0])
                    # If starts with 5-9 and has 4 digits, likely error
                    if first_digit >= 5:
                        integer_part = integer_part[1:]
                elif len(integer_part) >= 5:
                    # Definitely too long - remove first digit
                    integer_part = integer_part[1:]
                
                numbers_clean = f"{integer_part}.{decimal_part}" if decimal_part else integer_part
            else:
                if len(numbers) > 5:
                    numbers = numbers[1:]
                numbers_clean = numbers
            
            result = f"{province}-{letters} {numbers_clean}"
            return result
        
        # Fallback
        parts = text_clean.split()
        if len(parts) >= 2:
            first = parts[0]
            first = re.sub(r'^(\d{2})([A-Z])', r'\1-\2', first)
            parts[0] = first
            return ' '.join(parts)
        
        return text_clean
    
    def select_best_candidate(self, candidates):
        """Select best OCR result from multiple candidates"""
        # Sort by: valid format first, then confidence
        valid_candidates = [c for c in candidates if c[2]]
        
        if valid_candidates:
            # Choose valid candidate with highest confidence
            return max(valid_candidates, key=lambda x: x[1])
        else:
            # No valid candidates, choose highest confidence
            return max(candidates, key=lambda x: x[1])
    
    def validate_plate_format(self, text):
        """Validate if text matches Vietnamese license plate format"""
        import re
        
        if text == "N/A" or not text:
            return False
        
        # Remove extra spaces for validation
        text_clean = re.sub(r'\s+', ' ', text.strip())
        
        # Vietnamese plate patterns (more flexible):
        # XX-YZ NNNN.NN or XX-YZ NNNNN or XX-YZ-NNNN.NN (various formats)
        patterns = [
            r'^\d{2}-[A-Z]{1,2}\s+\d{3,5}\.?\d{0,2}$',      # 51-G 100.96
            r'^\d{2}-[A-Z]{1,2}-\d{3,5}\.?\d{0,2}$',        # 51-G-100.96
            r'^\d{2}[A-Z]{1,2}\s+\d{3,5}\.?\d{0,2}$',       # 51G 100.96
            r'^\d{2}-[A-Z]{1,2}\d{3,5}\.?\d{0,2}$',         # 51-G100.96
            r'^\d{1,2}\.[0-9A-Z]{1,2}-\d{3,5}\.?\d{0,2}$',  # Allow some OCR errors
        ]
        
        for pattern in patterns:
            if re.match(pattern, text_clean):
                return True
        
        # Partial match: has province code pattern (XX- or XX) and numbers
        if re.search(r'\d{2}', text_clean) and re.search(r'\d{3,5}', text_clean):
            return True  # Lenient validation
        
        return False
    
    def preprocess_for_lowres(self, cropped_plate):
        """Preprocessing optimized for low-resolution images"""
        h, w = cropped_plate.shape[:2]
        
        # Step 1: Upscale significantly (7x for very small plates)
        scale_factor = 7.0 if w < 200 else 6.0
        upscaled = cv2.resize(cropped_plate, None, fx=scale_factor, fy=scale_factor, 
                             interpolation=cv2.INTER_CUBIC)
        
        # Step 2: Denoise to reduce compression artifacts
        denoised = cv2.fastNlMeansDenoisingColored(upscaled, None, h=8, hColor=8, 
                                                     templateWindowSize=7, searchWindowSize=21)
        
        # Step 3: Convert to grayscale
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # Step 4: Adaptive histogram equalization for better contrast (stronger)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        enhanced = clahe.apply(gray)
        
        # Step 5: Double sharpening for better digit clarity
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 11,-1],  # More aggressive
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        
        # Step 6: Unsharp masking for additional sharpness
        gaussian_blur = cv2.GaussianBlur(sharpened, (0, 0), 2.0)
        unsharp = cv2.addWeighted(sharpened, 2.5, gaussian_blur, -1.5, 0)
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
        
        # Step 7: Bilateral filter to smooth while preserving edges
        bilateral = cv2.bilateralFilter(unsharp, 9, 75, 75)
        
        # Step 8: Morphological operations to enhance character edges
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(bilateral, cv2.MORPH_CLOSE, kernel)
        
        # Step 9: Multi-threshold approach for better digit separation
        # Try Otsu's method first
        _, otsu = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Also try adaptive threshold
        adaptive = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        
        # Combine both methods (take the better result)
        combined = cv2.bitwise_and(otsu, adaptive)
        
        # Convert back to BGR for PaddleOCR
        result = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def preprocess_aggressive(self, cropped_plate):
        """Alternative aggressive preprocessing for difficult cases"""
        h, w = cropped_plate.shape[:2]
        
        # Step 1: Super upscale (9x for very small plates)
        scale_factor = 9.0 if w < 100 else 8.0
        upscaled = cv2.resize(cropped_plate, None, fx=scale_factor, fy=scale_factor, 
                             interpolation=cv2.INTER_LANCZOS4)
        
        # Step 2: Convert to LAB color space for better contrast control
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel (very aggressive)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Step 3: Grayscale
        gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        # Step 4: Double sharpening for maximum clarity
        kernel_sharpen = np.array([[-1,-1,-1,-1,-1],
                                   [-1, 2, 2, 2,-1],
                                   [-1, 2, 13, 2,-1],
                                   [-1, 2, 2, 2,-1],
                                   [-1,-1,-1,-1,-1]]) / 9.0
        sharpened = cv2.filter2D(gray, -1, kernel_sharpen)
        
        # Step 5: Morphological operations to enhance character edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morph = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        
        # Step 6: Unsharp mask for additional sharpening
        gaussian = cv2.GaussianBlur(morph, (0, 0), 2.5)
        unsharp = cv2.addWeighted(morph, 2.5, gaussian, -1.5, 0)
        unsharp = np.clip(unsharp, 0, 255).astype(np.uint8)
        
        # Step 7: Multi-threshold for best digit separation
        # Otsu's method
        _, otsu = cv2.threshold(unsharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive threshold
        adaptive = cv2.adaptiveThreshold(unsharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 13, 3)
        
        # Combine both
        combined = cv2.bitwise_and(otsu, adaptive)
        
        # Convert back to BGR
        result = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def preprocess_super_resolution(self, cropped_plate):
        """Super-resolution preprocessing with advanced techniques"""
        h, w = cropped_plate.shape[:2]
        
        # Step 1: Extreme upscaling (9x for tiny plates to maximize detail)
        scale = 9.0 if w < 100 else 8.0 if w < 150 else 7.0
        upscaled = cv2.resize(cropped_plate, None, fx=scale, fy=scale, 
                             interpolation=cv2.INTER_LANCZOS4)  # Better quality interpolation
        
        # Step 2: Edge-preserving denoising
        denoised = cv2.bilateralFilter(upscaled, 9, 75, 75)
        
        # Step 3: Convert to grayscale
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        
        # Step 4: Adaptive contrast enhancement (stronger for small plates)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        contrast = clahe.apply(gray)
        
        # Step 5: Aggressive sharpening
        kernel_sharpen = np.array([[-1,-1,-1],
                                   [-1, 11,-1],
                                   [-1,-1,-1]])
        sharpened = cv2.filter2D(contrast, -1, kernel_sharpen)
        
        # Step 6: Edge enhancement
        # Sobel for edge detection
        sobelx = cv2.Sobel(sharpened, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(sharpened, cv2.CV_64F, 0, 1, ksize=3)
        edges = np.sqrt(sobelx**2 + sobely**2)
        edges = np.uint8(edges / edges.max() * 255)
        
        # Combine original with edges
        enhanced = cv2.addWeighted(sharpened, 0.8, edges, 0.2, 0)
        
        # Step 7: Adaptive binarization with multiple thresholds
        binary1 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        binary2 = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
        
        # Combine both thresholding methods
        combined = cv2.bitwise_and(binary1, binary2)
        
        # Step 8: Morphological cleanup
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Convert back to BGR
        result = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def process_ocr_result(self, ocr_result):
        """Process OCR result and return text with confidence"""
        if not ocr_result or not ocr_result[0]:
            return "N/A", 0.0
        
        result_dict = ocr_result[0]
        texts = []
        total_confidence = 0
        count = 0
        
        if 'rec_texts' in result_dict and 'rec_scores' in result_dict and 'dt_polys' in result_dict:
            for i in range(len(result_dict['rec_texts'])):
                text = result_dict['rec_texts'][i]
                conf = result_dict['rec_scores'][i]
                if conf > 0.3:  # Lower threshold to capture more text
                    bbox = result_dict['dt_polys'][i]
                    y_top = min(point[1] for point in bbox)
                    texts.append((y_top, text))
                    total_confidence += conf
                    count += 1
            
            texts.sort(key=lambda x: x[0])
            
            # Better logic for Vietnamese license plate ordering
            # Pattern should be: XX-YZ NNNN.NN (province-letters numbers)
            if len(texts) == 2:
                first_text = texts[0][1].strip()
                second_text = texts[1][1].strip()
                
                # Check if we need to swap based on Vietnamese license plate pattern
                # First apply OCR corrections to better identify the pattern
                corrected_first = self.fix_common_ocr_errors(first_text)
                corrected_second = self.fix_common_ocr_errors(second_text)
                
                # Check if text looks like a province code (XX-YZ format)
                first_looks_like_province = self.looks_like_province_code(corrected_first)
                second_looks_like_province = self.looks_like_province_code(corrected_second)
                
                # The part that's mostly numbers (with possible decimal) should come second
                first_is_mostly_numbers = self.is_mostly_numbers(first_text)
                second_is_mostly_numbers = self.is_mostly_numbers(second_text)
                
                # Priority-based ordering
                should_swap = False
                
                # Rule 1: If one looks like province code and other is mostly numbers, province first
                if second_looks_like_province and first_is_mostly_numbers and not first_looks_like_province:
                    should_swap = True
                # Rule 2: If first text is decimal numbers and second has dash, swap
                elif '.' in first_text and '-' in second_text:
                    should_swap = True
                # Rule 3: Fallback - if first is only digits and doesn't look like province
                elif (first_text.replace('.', '').replace(',', '').replace('-', '').isdigit() 
                      and not first_looks_like_province):
                    should_swap = True
                
                if should_swap:
                    texts = texts[::-1]
                    # Update the text values with corrected versions
                    texts[0] = (texts[0][0], corrected_second)
                    texts[1] = (texts[1][0], corrected_first)
                else:
                    # Apply corrections even if not swapping
                    texts[0] = (texts[0][0], corrected_first)
                    texts[1] = (texts[1][0], corrected_second)
            
            combined_text = ' '.join(t[1] for t in texts) if texts else "N/A"
            avg_confidence = total_confidence / count if count > 0 else 0.0
            
            return combined_text, avg_confidence
        
        return "N/A", 0.0
    
    def fix_common_ocr_errors(self, text):
        """Fix common OCR misreadings for Vietnamese license plates"""
        if text == "N/A" or not text:
            return text
        
        import re
        
        # Step 0: Fix special character confusions (Unicode issues)
        # À (A with grave accent) is often misread as -A
        text = text.replace('À', '-A')
        text = text.replace('à', '-a')
        # Other common Unicode confusions
        text = text.replace('Ì', '-I')
        text = text.replace('Ò', '-O')
        text = text.replace('È', '-E')
        
        # Step 1: Remove all extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Step 2: Fix common spacing/punctuation issues
        # Fix "51G -100.96" -> "51-G-100.96"
        text = re.sub(r'(\d{2})([A-Z])\s*-\s*(\d)', r'\1-\2-\3', text)
        # Fix "51 G 100.96" -> "51-G-100.96"
        text = re.sub(r'(\d{2})\s+([A-Z]{1,2})\s+(\d)', r'\1-\2 \3', text)
        # Fix "516-100.96" -> "51-G-100.96" (missing letter)
        text = re.sub(r'^(\d)(\d)(\d)-(\d)', r'\1\2-G-\3\4', text)
        # Fix decimal spacing: "100. 96" -> "100.96"
        text = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', text)
        # Fix "100 .96" -> "100.96"
        text = re.sub(r'(\d+)\s+\.(\d+)', r'\1.\2', text)
        # Fix dash variations: "51:G" -> "51-G", "51/G" -> "51-G"
        text = re.sub(r'(\d{2})[:/_·]([A-Z])', r'\1-\2', text)
        # Fix letter-number separator: "G:100" -> "G-100", "G/100" -> "G-100"
        text = re.sub(r'([A-Z]{1,2})[:/_=·](\d)', r'\1-\2', text)
        
        # Step 3: Common character confusions
        corrections = {
            'G': '6', 'g': '6',
            'O': '0', 'o': '0',
            'I': '1', 'l': '1',
            'S': '5', 's': '5',
            'B': '8',
            'D': '0',
            'Q': '0',
            'Z': '2',  # Sometimes Z is misread 2
        }
        
        # Step 4: Pattern-based correction
        parts = text.split(' ')
        
        if len(parts) >= 2:
            # First part: Province code (XX-YZ format)
            first_part = parts[0]
            
            if '-' in first_part:
                prefix_suffix = first_part.split('-', 1)
                if len(prefix_suffix) == 2:
                    prefix, suffix = prefix_suffix
                    
                    # Correct prefix (must be 2 digits)
                    corrected_prefix = ""
                    for char in prefix:
                        if char in ['O', 'o', 'D', 'Q']:
                            corrected_prefix += '0'
                        elif char in ['I', 'l']:
                            corrected_prefix += '1'
                        elif char in ['G', 'g']:
                            corrected_prefix += '6'
                        elif char in ['S', 's']:
                            corrected_prefix += '5'
                        elif char in ['B']:
                            corrected_prefix += '8'
                        elif char.isdigit():
                            corrected_prefix += char
                        else:
                            corrected_prefix += char
                    
                    # Ensure prefix is 2 digits
                    if len(corrected_prefix) == 1:
                        corrected_prefix = '5' + corrected_prefix  # Default to 5X
                    elif len(corrected_prefix) > 2:
                        corrected_prefix = corrected_prefix[:2]
                    
                    # Correct suffix (1-2 letters)
                    corrected_suffix = suffix.upper()
                    
                    # Fix common letter confusions in suffix
                    if corrected_suffix in ['21', 'Z1', 'ZI', '2I']:
                        corrected_suffix = 'Z1'  # L1 series
                    elif corrected_suffix == '100' or corrected_suffix == 'I00' or corrected_suffix == '10O':
                        corrected_suffix = 'T00'
                    else:
                        # First char should be letter
                        temp_suffix = ""
                        for i, char in enumerate(corrected_suffix):
                            if i == 0:  # First position - must be letter
                                if char == '6':
                                    temp_suffix += 'G'
                                elif char == '1':
                                    temp_suffix += 'T'
                                elif char == '0':
                                    temp_suffix += 'D'
                                elif char == '5':
                                    temp_suffix += 'S'
                                elif char == '8':
                                    temp_suffix += 'B'
                                elif char.isalpha():
                                    temp_suffix += char.upper()
                                else:
                                    temp_suffix += 'A'  # Default
                            else:  # Second position - letter or digit
                                if char in ['O', 'o', 'D', 'Q']:
                                    temp_suffix += '0'
                                elif char in ['I', 'l']:
                                    temp_suffix += '1'
                                elif char.isalnum():
                                    temp_suffix += char.upper()
                                else:
                                    temp_suffix += char
                        corrected_suffix = temp_suffix
                    
                    parts[0] = corrected_prefix + '-' + corrected_suffix
            else:
                # Missing dash - try to add it
                # Pattern: "51G" -> "51-G", "516100" -> "51-G-100"
                match = re.match(r'^(\d{2})([A-Z]{1,2})$', first_part.upper())
                if match:
                    parts[0] = match.group(1) + '-' + match.group(2)
            
            # Second part: Number (3-5 digits with optional decimal)
            if len(parts) >= 2:
                number_part = parts[-1]
                corrected_number = ""
                
                for char in number_part:
                    if char == '.':
                        corrected_number += char
                    elif char in ['O', 'o', 'D', 'Q']:
                        corrected_number += '0'
                    elif char in ['I', 'l']:
                        corrected_number += '1'
                    elif char in ['G', 'g']:
                        corrected_number += '6'
                    elif char in ['S', 's']:
                        corrected_number += '5'
                    elif char in ['B']:
                        corrected_number += '8'
                    elif char in ['Z', 'z']:
                        corrected_number += '2'
                    elif char.isdigit():
                        corrected_number += char
                    # Remove non-alphanumeric except dot
                    elif char not in ['-', '/', ':', ' ', '%']:
                        pass
                
                # Remove trailing non-digits
                corrected_number = re.sub(r'[^\d.]+$', '', corrected_number)
                
                parts[-1] = corrected_number
            
            text = ' '.join(parts)
        
        # Step 5: Final validation and cleanup
        # Ensure format: XX-YZ NNNNN.NN
        match = re.match(r'^(\d{2})-?([A-Z]{1,2})\s+(\d{3,5}\.?\d{0,2})$', text)
        if match:
            text = f"{match.group(1)}-{match.group(2)} {match.group(3)}"
        
        # Remove any remaining invalid characters
        text = re.sub(r'[%_\\/]', '', text)
        
        return text.strip()
    
    def is_mostly_numbers(self, text):
        """Check if text is mostly numbers (allowing dots and common OCR artifacts)"""
        if not text:
            return False
        
        # Remove common non-alphanumeric characters
        clean_text = text.replace('.', '').replace(',', '').replace(' ', '').replace('-', '')
        if not clean_text:
            return False
        
        # Count digits vs total characters
        digit_count = sum(1 for c in clean_text if c.isdigit())
        return digit_count / len(clean_text) > 0.7  # 70% or more digits
    
    def looks_like_province_code(self, text):
        """Check if text looks like a Vietnamese province code (XX-YZ format)"""
        if not text or '-' not in text:
            return False
        
        parts = text.split('-')
        if len(parts) != 2:
            return False
        
        prefix, suffix = parts
        
        # Prefix should be 2 digits (province number)
        if len(prefix) != 2 or not prefix.isdigit():
            return False
        
        # Suffix should be 1-2 characters (letter(s) or corrected from digits)
        if len(suffix) < 1 or len(suffix) > 2:
            return False
        
        # Vietnamese province codes typically are 10-99
        try:
            province_num = int(prefix)
            if province_num < 10 or province_num > 99:
                return False
        except ValueError:
            return False
        
        return True

    def draw_bboxes(self, frame, licence_plate_detections, licence_plate_texts):
        for bbox, text in zip(licence_plate_detections, licence_plate_texts):
            x1, y1, x2, y2 = map(int, bbox)
            
            # Remove confidence from displayed text
            display_text = text
            if "(confidence:" in text:
                display_text = text.split("(confidence:")[0].strip()
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{display_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        return frame