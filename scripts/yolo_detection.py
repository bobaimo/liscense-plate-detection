# Python
import cv2
import os
from pathlib import Path
import easyocr
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/aimo/models/license_plate.pt")

# Open the video file
# video_path = "hkstp_high.mp4"
video_path = "videos/chinese_clip.mp4"
cap = cv2.VideoCapture(video_path)

# Create output directory based on video name
video_name = Path(video_path).stem
output_dir = Path("cropped_box") / video_name
output_dir.mkdir(parents=True, exist_ok=True)

# Track saved objects to avoid duplicates
saved_track_ids = set()

# Track frame count for each track ID
track_frame_counts = {}
track_data = {}  # Store last seen data for each track
FRAME_THRESHOLD = 15

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLO11 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, conf=0.5)

        # Process detections and crop boxes
        current_frame_tracks = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                current_frame_tracks.add(track_id)
                
                # Initialize or increment frame count for this track
                if track_id not in track_frame_counts:
                    track_frame_counts[track_id] = 1
                else:
                    track_frame_counts[track_id] += 1
                
                # Store the current detection data
                track_data[track_id] = {
                    'box': box,
                    'class': cls,
                    'confidence': conf,
                    'frame': frame.copy()
                }
                
                # Save only if track has been seen for more than FRAME_THRESHOLD frames
                # and hasn't been saved yet
                if (track_frame_counts[track_id] > FRAME_THRESHOLD and 
                    track_id not in saved_track_ids):
                    
                    # Create a copy of the full frame for visualization
                    frame_with_bbox = frame.copy()
                    
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box)
                    bbox_size=abs(x1-x2)*abs(y1-y2)
                    if bbox_size<2000:
                        continue

                    crop_image=frame_with_bbox[y1:y2,x1:x2]
                    
                    # Draw bounding boxhkstp_high
                    # cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Create filename with track_id, class, and confidence
                    filename_with_bbox = f"track_{track_id}_with_box.jpg"
                    filename=f"track_{track_id}.jpg"
                    save_path_with_bbox = output_dir / filename_with_bbox
                    save_path = output_dir / filename


                    
                    # Save the full frame with bounding box
                    cv2.imwrite(str(save_path_with_bbox), crop_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                    
                    # Mark this track_id as saved
                    saved_track_ids.add(track_id)
                    
                    print(f"Saved: {filename} (after {track_frame_counts[track_id]} frames)")
        
        # Reset frame count for tracks not seen in current frame
        tracks_to_remove = []
        for track_id in track_frame_counts:
            if track_id not in current_frame_tracks:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del track_frame_counts[track_id]
            if track_id in track_data:
                del track_data[track_id]

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()