from easyocr_reader import easyocr_reader
from trocr_reader import trocr_reader
from pathlib import Path
import cv2
from ultralytics import YOLO
import csv
from LPRNet_reader import LPRNetInference
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import time
import threading
from queue import Queue



class rtsp_realtime_detection:
    def __init__(self, rtsp_url, license_type="hongkong"):
        self.rtsp_url = rtsp_url
        self.type = license_type
        if self.type == "hongkong":
            self.easyocr = easyocr_reader()
            self.trocr = trocr_reader()
            self.id_dict_easyocr, self.id_dict_trocr = {}, {}
        elif self.type == "china":
            self.LPRNet = LPRNetInference()
            self.id_dict_net = {}
        self.valid_result = {}
        
        self.detector = YOLO("./models/license_plate_11x.pt")
        self.confirmed_track_ids = set()
        self.FRAME_THRESHOLD = 10
        
        # Performance optimization settings
        self.frame_skip = 5  # Process every nth frame
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Threading for OCR processing
        self.ocr_queue = Queue(maxsize=10)
        self.ocr_results = {}
        self.ocr_thread_running = True
        self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
        self.ocr_thread.start()
        
        # Threading for result processing
        self.result_queue = Queue(maxsize=20)
        self.result_thread_running = True
        self.result_thread = threading.Thread(target=self._ocr_result_worker, daemon=True)
        self.result_thread.start()

        # Detection counter
        self.current_detections = 0

        # Create output directory based on RTSP URL
        stream_name = self.rtsp_url.replace("://", "_").replace("/", "_").replace(":", "_")
        self.output_dir = Path("cropped_box") / f"rtsp_stream_{stream_name}_{int(time.time())}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RTSP stream with timeout settings
        self.video_cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        
        # Optimize buffer settings for low latency and set timeout
        self.video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video_cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS
        # Set RTSP timeout to 60 seconds (in milliseconds)
        self.video_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
        self.video_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)
        # Additional optimizations
        self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not self.video_cap.isOpened():
            raise Exception(f"Failed to open RTSP stream: {rtsp_url}")
        
        self.csv_init()

    def csv_init(self):
        timestamp = int(time.time())
        self.csv_file_path = self.output_dir / f"license_detections_rtsp_{timestamp}.csv"
        self.csv_file = open(self.csv_file_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        if self.type == "hongkong":
            self.csv_writer.writerow(['Track_ID', 'Image_Path', 'License_Number_EASYOCR', 'License_Number_TROCR', 'Timestamp'])
        elif self.type == "china":
            self.csv_writer.writerow(['Track_ID', 'Image_Path', 'License_Number', 'Timestamp'])
    
    def _ocr_worker(self):
        """Background thread for OCR processing"""
        while self.ocr_thread_running:
            try:
                if not self.ocr_queue.empty():
                    task = self.ocr_queue.get(timeout=0.1)
                    if task is None:  # Shutdown signal
                        self.result_queue.put(None)
                        self.ocr_thread_running=False
                        break
                    
                    track_id, frame_crop,frame = task

                    match self.type:
                        case "hongkong":
                            license_num_easy, bbox_plate = self.easyocr.read_plate(frame_crop)
                            license_num_easy = license_num_easy.upper().replace('I', '1').replace('O', '0').replace('Q', '0')

                            # Process with TrOCR
                            license_num_trocr, _ = self.trocr.read_plate(frame_crop, bbox_plate)
                            license_num_trocr = license_num_trocr.upper().replace('I', '1').replace('O', '0').replace('Q', '0')

                            # Store results
                            self.result_queue.put((track_id, license_num_easy, license_num_trocr, frame))

                        case "china":
                            pass

                        case _:
                            print("region not specified")
                        

                    self.ocr_queue.task_done()
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"OCR worker error: {e}")
                continue

    def _ocr_result_worker(self):
        while self.result_thread_running:
            try:
                if not self.result_queue.empty():
                    result = self.result_queue.get(timeout=0.1)
                    if result is None:  # Shutdown signal
                        self.result_thread_running=False
                        break
                    track_id, license_num_easy,license_num_trocr,frame = result

                    match self.type:
                        case "hongkong":
                            if track_id not in self.confirmed_track_ids:
                                # Initialize valid_result for this track_id if not exists
                                if track_id not in self.valid_result:
                                    self.valid_result[track_id] = {"easyocr": None, "trocr": None}

                                if license_num_easy != "":
                                    if track_id not in self.id_dict_easyocr:
                                        self.id_dict_easyocr[track_id] = {}
                                    if license_num_easy not in self.id_dict_easyocr[track_id]:
                                        self.id_dict_easyocr[track_id][license_num_easy] = 0
                                    self.id_dict_easyocr[track_id][license_num_easy] += 1
                                    if self.id_dict_easyocr[track_id][license_num_easy] > self.FRAME_THRESHOLD:
                                        self.valid_result[track_id]["easyocr"] = license_num_easy
                                        save_path = self.out_dir/f"{str(self.easyocr)}_{track_id}_put{license_num_easy}.jpg"
                                        cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                                        print(f"easyocr detection successful with number{license_num_easy}")
                                
                                if license_num_trocr != "":
                                    if track_id not in self.id_dict_trocr:
                                        self.id_dict_trocr[track_id] = {}
                                    if license_num_trocr not in self.id_dict_trocr[track_id]:
                                        self.id_dict_trocr[track_id][license_num_trocr] = 0
                                    self.id_dict_trocr[track_id][license_num_trocr] += 1
                                    if self.id_dict_trocr[track_id][license_num_trocr] > self.FRAME_THRESHOLD:
                                        self.valid_result[track_id]["trocr"] = license_num_trocr
                                        save_path = self.output_dir/f"{str(self.trocr)}_{track_id}_{license_num_trocr}.jpg"
                                        cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                                        print(f"trocr detection successful with number{license_num_trocr}")
                                
                                # Check if both OCR methods agree
                                if (self.valid_result[track_id]["easyocr"] is not None and 
                                    self.valid_result[track_id]["trocr"] is not None and
                                    self.valid_result[track_id]["easyocr"] == self.valid_result[track_id]["trocr"]):
                                    self.confirmed_track_ids.add(track_id)
                                    # Write to CSV
                        case "china":
                            pass

                        case _:
                            print("region not specified")

                    self.result_queue.task_done()
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"OCR worker error: {e}")
                continue

    def detection(self, frame):
        self.frame_count += 1
        results = self.detector.track(frame, persist=True, conf=0.6, verbose=False)     
        if results[0].boxes and results[0].boxes.id is not None:
            xyxys = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            # Count current detections
            self.current_detections = len(track_ids)
            
            for xyxy, track_id in zip(xyxys, track_ids):
                x1, y1, x2, y2 = map(int, xyxy)
                
                # Process OCR on every nth frame for performance
                if self.frame_count % self.frame_skip == 0:
                    match self.type:
                        case "hongkong":
                            if track_id not in self.confirmed_track_ids:
                                if not self.ocr_queue.full():
                                    license_image = frame[y1:y2, x1:x2].copy()
                                    self.ocr_queue.put((track_id, license_image, frame.copy()))

                        case "china":
                            frame = self.reader_logging_net(track_id, frame, x1, y1, x2, y2, time.time())

                        case _:
                            print("no region specified")
       
                # Always display bounding boxes and license plates on every frame
                if track_id in self.confirmed_track_ids:
                    frame = self.visualization(frame, self.valid_result[track_id]["easyocr"], x1, y1, x2, y2)
                else:
                    # Show bounding box without text if not yet detected
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
        # Show FPS and detection count
        frame = self._display_info(frame)
        # Display realtime output
        cv2.imshow("RTSP License Detection", frame)
        return frame
    
    def _display_info(self, frame):        
        # Display detection count in top right corner
        frame_width = frame.shape[1]
        detection_text = f"Detections: {self.current_detections}"
        text_size = cv2.getTextSize(detection_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        cv2.putText(frame, detection_text, (frame_width - text_size[0] - 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Resize frame for display (scale down large frames)
        max_display_width = 1280
        max_display_height = 720
        height, width = frame.shape[:2]
        
        # Calculate scaling factor
        scale_width = max_display_width / width
        scale_height = max_display_height / height
        scale = min(scale_width, scale_height, 1.0)  # Don't scale up
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_frame = cv2.resize(frame, (new_width, new_height))
        else:
            resized_frame = frame
            
        return resized_frame

    def visualization(self, frame, label, x1, y1, x2, y2):
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
                        break
                    except Exception:
                        continue
                if font is None:
                    font = ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            
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

    def reader_logging_net(self, track_id, frame, x1, y1, x2, y2, timestamp):
        if track_id not in self.id_final_net:
            license_image = frame[y1:y2, x1:x2]
            license_num = self.LPRNet.predict(license_image)
            if track_id not in self.id_dict_net:
                self.id_dict_net[track_id] = {}
            if license_num not in self.id_dict_net[track_id]:
                self.id_dict_net[track_id][license_num] = 0
            self.id_dict_net[track_id][license_num] += 1
            if self.id_dict_net[track_id][license_num] > self.FRAME_THRESHOLD:
                self.id_final_net[track_id] = license_num
                save_path = self.output_dir / f"{str(self.LPRNet)}_{track_id}_{license_num}_{int(timestamp)}.jpg"
                cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
                self.csv_writer.writerow([track_id, f"{self.output_dir}/{str(self.LPRNet)}_{track_id}_{license_num}_{int(timestamp)}.jpg",
                                    license_num, timestamp])
        else:
            frame = self.visualization(frame, self.id_final_net[track_id], x1, y1, x2, y2)
        return frame
    

    def write_csv(self):
        """Write all confirmed detections to CSV file"""
        for track_id in self.confirmed_track_ids:
            if track_id in self.valid_result:
                easyocr_result = self.valid_result[track_id].get("easyocr", "")
                trocr_result = self.valid_result[track_id].get("trocr", "")
                timestamp = time.time()
                
                # Write to CSV
                self.csv_writer.writerow([
                    track_id,
                    f"{self.output_dir}/confirmed_{track_id}_{easyocr_result}.jpg",
                    easyocr_result,
                    trocr_result,
                    timestamp
                ])
        
        self.csv_file.flush()

    def main(self):
        print(f"Starting RTSP stream processing: {self.rtsp_url}")
        print("Press 'q' to quit")
        
        try:
            while True:
                success, frame = self.video_cap.read()
                if success:
                    self.detection(frame)
                    # Reduced wait time for better performance
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Failed to read frame from RTSP stream")
                    # Try to reconnect with proper settings
                    self.video_cap.release()
                    time.sleep(1)
                    self.video_cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                    self.video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.video_cap.set(cv2.CAP_PROP_FPS, 15)
                    self.video_cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 60000)
                    self.video_cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 60000)
                    self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    if not self.video_cap.isOpened():
                        print("Failed to reconnect to RTSP stream")
                        break
        except KeyboardInterrupt:
            print("\nStopping RTSP stream processing...")
        finally:
            # Cleanup
            self.ocr_queue.put(None)
            while self.result_thread_running is not False:
                time.sleep(0.1)
            self.write_csv()
            self.video_cap.release()
            self.csv_file.close()
            cv2.destroyAllWindows()
            print(f"License detections saved to {self.csv_file_path}")


if __name__ == "__main__":
    rtsp_url="rtsp://admin:rsxx1111@192.168.10.175/stream1"
    # rtsp_url="rtsp://localhost:8554/"
    try:
        # Choose license type: "hongkong" or "china"
        detector = rtsp_realtime_detection(rtsp_url, license_type="hongkong")
        detector.main()
    except Exception as e:
        print(f"Error processing RTSP stream {rtsp_url}: {e}")