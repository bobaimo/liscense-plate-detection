from easyocr_reader import easyocr_reader
from trocr_reader import trocr_reader
from pathlib import Path
import cv2
from ultralytics import YOLO
import csv
from LPRNet_reader import LPRNetInference
from PIL import Image, ImageDraw, ImageFont
import numpy as np



class realtime_detection:
    def __init__(self,video_path,license_type="hongkong"):
        self.type=license_type
        if self.type=="hongkong":
            self.easyocr=easyocr_reader()
            self.trocr=trocr_reader()
            self.id_dict_easyocr={}
            self.id_dict_trocr={}
        elif self.type=="china":
            self.LPRNet=LPRNetInference()
            self.id_dict_net={}
        self.valid_result={}
        
        self.detector=YOLO("./models/license_plate_11x.pt")
        self.confirmed_track_ids=set()
        self.FRAME_THRESHOLD = 15

        self.output_dir=Path("cropped_box") / Path(video_path).stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.video_cap=cv2.VideoCapture(video_path)

        self.output_init(video_path)
        self.csv_init(video_path)

        
# Initialize CSV file for detections

    def csv_init(self,video_path):
        self.csv_file_path = self.output_dir / f"license_detections_{Path(video_path).stem}.csv"
        self.csv_file = open(self.csv_file_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        if self.type=="hongkong":
            self.csv_writer.writerow(['Track_ID', 'Image_Path', 'License_Number_EASYOCR', 'License_Number_TROCR'])
        elif self.type=="china":
            self.csv_writer.writerow(['Track_ID', 'Image_Path', 'License_Number'])


    def output_init(self,video_path):
        fps = 30  # Set output video to 30 fps
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output_video_path=self.output_dir/f"{Path(video_path).stem}_output.mp4"
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))

    def detection(self,frame):
        bbox_list=[]
        results = self.detector.track(frame, persist=True, conf=0.6)
        if results[0].boxes and results[0].boxes.id is not None:
            xyxys = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            for xyxy, track_id in zip(xyxys, track_ids):
                x1, y1, x2, y2 = map(int, xyxy)
                if track_id not in self.confirmed_track_ids:    
                    if self.type=="hongkong":
                        crop_frame=frame[y1:y2,x1:x2]
                        license_num_easy,bbox_plate=self.easyocr.read_plate(crop_frame)
                        license_num_trocr=self.trocr.read_plate(crop_frame,bbox_plate)
                        self.reader_logging_ocr(track_id,[license_num_easy,license_num_trocr],frame.copy())
                        if track_id in self.confirmed_track_ids:
                            frame=self.visualization(frame,self.valid_result[track_id]["easyocr"],x1, y1, x2, y2)
                        else:
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    elif self.type=="china":
                        x1, y1, x2, y2 = map(int, xyxy)
                        frame=self.reader_logging_net(track_id,frame,x1, y1, x2, y2)    
        self.out.write(frame)
        
        # Resize frame to fit monitor
        screen_height = 720  # Adjust based on your monitor
        height, width = frame.shape[:2]
        if height > screen_height:
            scale = screen_height / height
            new_width = int(width * scale)
            resized_frame = cv2.resize(frame, (new_width, screen_height))
        else:
            resized_frame = frame
        
        cv2.imshow("YOLO11 Tracking", resized_frame)
        cv2.waitKey(10)

    def visualization(self,frame,label,x1, y1, x2, y2):
        if self.type == "china":
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(pil_image)
            
            # Try to load a Chinese font, fallback to default if not available
            try:
                # Common Chinese font paths on Linux
                font_paths = [
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                ]
                font = None
                for font_path in font_paths:
                    try:
                        font = ImageFont.truetype(font_path, 30)
                        print(f"Successfully loaded font: {font_path}")
                        break
                    except Exception as e:
                        print(f"Failed to load {font_path}: {e}")
                        continue
                if font is None:
                    font = ImageFont.load_default()
                    print("Using default font")
            except:
                font = ImageFont.load_default()
                print("Using default font (exception)")
            
            # Draw rectangle
            draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=2)
            
            # Get text size
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Draw background rectangle for text
            draw.rectangle([(x1, y1-text_height-20), (x1+text_width+10, y1-5)], fill=(255, 0, 0))
            
            # Draw text
            draw.text((x1+5, y1-text_height-15), label, font=font, fill=(255, 255, 255))
            
            # Convert back to BGR for OpenCV
            frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            return frame_bgr
        else:
            # Original visualization for non-Chinese types
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 3)[0]
            cv2.rectangle(frame, (x1, y1-text_size[1]-20), (x1+text_size[0]+10, y1-5), (255, 0, 0), -1)
            cv2.putText(frame, label, (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
            return frame

    def reader_logging_ocr(self,track_id,license_list,frame):
        for license, id_dict, reader in zip(license_list,[self.id_dict_easyocr,self.id_dict_trocr],[self.easyocr,self.trocr]):
            if license != "":
                license = str(license).upper().replace('I', '1').replace('O', '0').replace('Q', '0')
                if track_id not in id_dict:
                    id_dict[track_id]={}
                if license not in id_dict[track_id]:
                    id_dict[track_id][license]=0
                id_dict[track_id][license]+=1
                if id_dict[track_id][license]> self.FRAME_THRESHOLD:
                    if track_id not in self.valid_result:
                        self.valid_result[track_id] = {"easyocr": None, "trocr": None}
                    self.valid_result[track_id][str(reader)]=license
                    save_path=self.output_dir/Path(str(reader))/f"{track_id}_{license}.jpg"
                    save_path.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(f"{str(save_path)}/{track_id}_{license}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                    print(f"{str(reader)} detection successful with number{license}")
                    if self.valid_result[track_id]["easyocr"] == self.valid_result[track_id]["trocr"]:
                        self.confirmed_track_ids.add(track_id)

    def reader_logging_net(self,track_id,frame,x1,y1,x2,y2):
        if track_id not in self.id_final_net:
            license_image=frame[y1:y2,x1:x2]
            # if self.is_blurry(license_image):
            #     return frame
            license_num=self.LPRNet.predict(license_image)
            if track_id not in self.id_dict_net:
                self.id_dict_net[track_id] = {}
            if license_num not in self.id_dict_net[track_id]:
                self.id_dict_net[track_id][license_num] = 0
            self.id_dict_net[track_id][license_num] += 1
            if self.id_dict_net[track_id][license_num]>self.FRAME_THRESHOLD:
                self.id_final_net[track_id]=license_num
                save_path = self.output_dir/f"{str(self.LPRNet)}_{track_id}_{license_num}.jpg"
                cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                self.csv_writer.writerow([track_id,f"{self.output_dir}/{str(self.LPRNet)}_{track_id}_{license_num}.jpg",
                                    license_num])
        else:
            frame=self.visualization(frame,self.id_final_net[track_id],x1, y1, x2, y2)
        return frame

    def write_csv(self):
        for track_id in self.valid_result:
            easyocr_result = self.valid_result[track_id].get("easyocr", "")
            trocr_result = self.valid_result[track_id].get("trocr", "")
            if track_id in self.confirmed_track_ids or easyocr_result is not None:
                save_dir=f"{self.output_dir}/easyocr/{track_id}_{easyocr_result}.jpg"
            else:
                save_dir=f"{self.output_dir}/trocr/{track_id}_{trocr_result}.jpg"

            # Write to CSV
            self.csv_writer.writerow([
                track_id,
                save_dir,
                easyocr_result,
                trocr_result
            ])

    def main(self):
        while self.video_cap.isOpened():
            success, frame = self.video_cap.read()
            if success:
                self.detection(frame)
            else:
                break

        self.writecsv()
        self.out.release()
        self.video_cap.release()
        self.csv_file.close()  # Close CSV file
        cv2.destroyAllWindows()
        print(f"Video saved to {self.output_video_pavideo_pathth}")
        print(f"License detections saved to {self.csv_file_path}")


    

if __name__=="__main__":
    # video_list=["carpark_1.mp4","carpark_2.mp4","hkstp_trial.mp4","hkstp_high.mp4","cropped_videoplayback.mp4","front_left_107.mp4"]
    video_list=["./videos/infra_red_carpark_1_rotated.mp4"]
    # video_list=["./videos/only_wp.mp4"]

    # video_list=["./videos/china_carpark_1.mp4"]
    for video_path in video_list:
        # d=realtime_detection(video_path,license_type="china")
        d=realtime_detection(video_path,license_type="hongkong")
        d.main()


