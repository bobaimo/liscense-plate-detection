from read_license_plate import ocr_reader
from pathlib import Path
import cv2
from ultralytics import YOLO

#Initialization
video_path="hkstp_high.mp4"
output_dir=Path("cropped_box") / Path(video_path).stem
output_dir.mkdir(parents=True, exist_ok=True)
FRAME_THRESHOLD = 15
r=ocr_reader()
detector=YOLO("best.pt")
video_cap=cv2.VideoCapture(Path("sample") /video_path)
id_dict={}
id_final={}

# Setup video writer for saving output
fps = video_cap.get(cv2.CAP_PROP_FPS)
width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video_path=Path("sample")/'output_video.mp4'
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


def detection(frame):
    results = detector.track(frame, persist=True, conf=0.5)
    if results[0].boxes and results[0].boxes.id is not None:
        xyxys = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy().astype(int)
        for xyxy, track_id in zip(xyxys, track_ids):
            x1, y1, x2, y2 = map(int, xyxy)
            if track_id not in id_final:
                label=""
                license_image=frame[y1:y2,x1:x2]
                license_num=r.read_plate(license_image)
                if track_id not in id_dict:
                    id_dict[track_id] = {}
                if license_num not in id_dict[track_id]:
                    id_dict[track_id][license_num] = 0
                id_dict[track_id][license_num] += 1
                if id_dict[track_id][license_num]>10:
                    id_final[track_id]=license_num
                    save_path = output_dir/f"track_{track_id}_{license_num}.jpg"
                    cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    label=license_num
            else:
                label=id_final[track_id]
            frame=visualization(frame,label,x1, y1, x2, y2)
    out.write(frame)
    cv2.imshow("YOLO11 Tracking", frame)
    cv2.waitKey(10)

def visualization(frame,label,x1, y1, x2, y2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
    cv2.rectangle(frame, (x1, y1-text_size[1]-20), (x1+text_size[0]+10, y1-5), (255, 0, 0), -1)
    cv2.putText(frame, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return frame

def main():
    while video_cap.isOpened():
        success, frame = video_cap.read()
        if success:
            detection(frame)
        else:
            break
    out.release()
    video_cap.release()
    cv2.destroyAllWindows()
    print("Video saved as output_video.mp4")


    

if __name__=="__main__":
    main()


