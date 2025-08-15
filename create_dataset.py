from pathlib import Path
import cv2
from ultralytics import YOLO
from visual_llm_detection import visual_llm

# Initialization
detector = YOLO("best.pt")
vlm=visual_llm()
video_list=["carpark_1.mp4","carpark_2.mp4","hkstp_trial.mp4","hkstp_high.mp4","cropped_videoplayback.mp4","front_left_107.mp4"]

def create_dir(video_path):
    raw_image_dir = Path("dataset")/Path(video_path).stem/Path("raw_image")
    label_dir = Path("dataset")/Path(video_path).stem/Path("label")
    bbox_dir = Path("dataset")/Path(video_path).stem/Path("bounding_box")
    raw_image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    bbox_dir.mkdir(parents=True, exist_ok=True)
    return raw_image_dir, label_dir, bbox_dir
def create_dataset_from_video(video_path):
    raw_image_dir,label_dir,bbox_dir=create_dir(video_path)
    video_cap = cv2.VideoCapture(Path("videos") / video_path)
    frame_count = 0
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if frame_count%15!=0:
            frame_count+=1
            continue
        if success:
            results = detector.track(frame, persist=True, conf=0.5)
            if results[0].boxes and results[0].boxes.id is not None:
                # Save frame only when detections exist
                frame_save_path = raw_image_dir / f"frame_{frame_count:06d}_{Path(video_path).stem}.jpg"
                label_save_path = label_dir / f"frame_{frame_count:06d}.txt"
                cv2.imwrite(str(frame_save_path), frame)
                
                # Get frame dimensions for YOLO normalization
                frame_height, frame_width = frame.shape[:2]
                
                # Prepare YOLO format labels
                yolo_labels = []
                
                xyxys = results[0].boxes.xyxy.cpu().numpy()
                for bbox_idx, xyxy in enumerate(xyxys):
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    # Crop bounding box from frame
                    cropped_bbox = frame[y1:y2, x1:x2]

                    if vlm.prompt_detection(cropped_bbox,"Is the image a car license plate? Answer only 'yes' or 'no'."):
                        # Convert to YOLO format (normalized center_x, center_y, width, height)
                        center_x = (x1 + x2) / 2 / frame_width
                        center_y = (y1 + y2) / 2 / frame_height
                        bbox_width = (x2 - x1) / frame_width
                        bbox_height = (y2 - y1) / frame_height

                        # Add to YOLO labels (class 0 for license plate)
                        yolo_labels.append(f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")
                        
                        bbox_save_path = bbox_dir / f"frame_{frame_count:06d}_{len(yolo_labels)}.jpg"
                        cv2.imwrite(str(bbox_save_path), cropped_bbox)
                        # Convert to YOLO format (normalized center_x, center_y, width, height)
                
                # Write YOLO format labels to file
                with open(label_save_path, 'w') as f:
                    f.write('\n'.join(yolo_labels))
                
                print(f"Saved frame {frame_count} with {len(yolo_labels)} detections")
                frame_count += 1
        else:
            break
    
    video_cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset creation complete! Saved {frame_count} frames with detections.")

if __name__ == "__main__":
    for video_path in video_list:
        create_dataset_from_video(video_path)