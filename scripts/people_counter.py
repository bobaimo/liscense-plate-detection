from ultralytics import YOLO
import cv2
from pathlib import Path

class people_counter:

    def __init__(self,model="./models/yolo11x.pt"):
        self.detector=YOLO(model)

    def count(self,frame):
        results = self.detector.track(frame, persist=True, conf=0.4, classes=[0])
        if results[0].boxes and results[0].boxes.id is not None:
            xyxys = results[0].boxes.xyxy.cpu().numpy()
        return len(xyxys),xyxys
    
    def visualize(self,frame,count,bboxes,out=None):
        screen_width = 1700  # Adjust based on your monitor
        height, width = frame.shape[:2]
        
        # Draw bounding boxes on original frame before resizing
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add people count text to original frame before resizing
        cv2.putText(frame, f"PEOPLE COUNT: {count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Write original frame to video (before resizing for display)
        if out is not None:
            out.write(frame)
        
        # Resize for display only
        if width > screen_width:
            scale = screen_width / width
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (screen_width, new_height))
        else:
            resized_frame = frame

        cv2.imshow("PEOPLE COUNT", resized_frame)
        cv2.waitKey(1)
            
        


if __name__=="__main__":
    p=people_counter()
    video_path="./videos/cctv.mp4"
    video_cap=cv2.VideoCapture(video_path)
    output_dir=Path("people_counter") / Path(video_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path=output_dir/f"{Path(video_path).stem}_output.mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, 30,(width, height))
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if success:
            count,bboxes=p.count(frame)
            p.visualize(frame,count,bboxes,out)
        else:
            break
    out.release()
    video_cap.release()
    cv2.destroyAllWindows()

