import cv2
from ultralytics import YOLO
import os  # Để check file


class VehicleDetection:
    def __init__(self, model_path):
        onnx_path = model_path.replace('.pt', '.onnx')

        # Nếu chưa có file ONNX thì export từ file .pt
        if not os.path.exists(onnx_path):
            model = YOLO(model_path)
            model.export(format='onnx')

        # Sử dụng ONNX cho tốc độ nhanh hơn
        self.model = YOLO(onnx_path, task='detect')

    def detect_frame(self, frame, frame_count):
        tracker_config = {"persist": True}

        # Cứ mỗi 10 frame thì giảm IOU để tăng tốc
        if frame_count % 10 == 0:
            results = self.model.track(frame, iou=0.1, conf=0.30, **tracker_config)
        else:
            results = self.model.track(frame, **tracker_config)

        return self.process_results(results[0])

    def process_results(self, results):
        id_name_dict = results.names
        vehicle_list = []

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            cls_id = int(box.cls.tolist()[0])
            cls_name = id_name_dict[cls_id]

            if cls_name in ["car", "motorcycle"]:
                vehicle_list.append((result, cls_name))

        return vehicle_list

    def draw_bboxes(self, frame, vehicle_detections):
        for idx, (bbox, cls_name) in enumerate(vehicle_detections):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.putText(
                frame,
                f"{cls_name.capitalize()}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        return frame
