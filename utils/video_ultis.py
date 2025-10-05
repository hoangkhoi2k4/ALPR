# Import all the Required Libraries
import cv2

# Read the Video with frame sampling
def read_video(video_path, frame_step=2):  # Default step=2: xử lý 1/2 frames, tăng lên 3-5 nếu cần nhanh hơn
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_step == 0:
            frames.append(frame)
        frame_count += 1
    cap.release()
    return frames

# Save the Video
def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()