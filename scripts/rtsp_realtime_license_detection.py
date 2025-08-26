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
            self.id_final_easyocr, self.id_final_trocr = {}, {}
        elif self.type == "china":
            self.LPRNet = LPRNetInference()
            self.id_dict_net, self.id_final_net = {}, {}
        
        self.detector = YOLO("./models/license_plate_11x.pt")
        self.written_track_ids = set()
        self.FRAME_THRESHOLD = 15
        
        # Performance optimization settings
        self.frame_skip = 2  # Process every nth frame
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # Threading for OCR processing
        self.ocr_queue = Queue(maxsize=10)
        self.ocr_results = {}
        self.ocr_thread_running = True
        if self.type == "hongkong":
            self.ocr_thread = threading.Thread(target=self._ocr_worker, daemon=True)
            self.ocr_thread.start()

        # Create output directory based on RTSP URL
        stream_name = self.rtsp_url.replace("://", "_").replace("/", "_").replace(":", "_")
        self.output_dir = Path("cropped_box") / f"rtsp_stream_{stream_name}_{int(time.time())}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize RTSP stream
        self.video_cap = cv2.VideoCapture(rtsp_url)
        # Optimize buffer settings for low latency
        self.video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video_cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS
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
                        break
                    
                    track_id, frame_crop, timestamp, x1, y1, x2, y2 = task
                    
                    # Process with EasyOCR
                    license_num_easy, _ = self.easyocr.read_plate(frame_crop, [])
                    license_num_easy = license_num_easy.upper().replace('I', '1').replace('O', '0').replace('Q', '0')
                    
                    # Process with TrOCR
                    license_num_trocr, _ = self.trocr.read_plate(frame_crop, [])
                    license_num_trocr = license_num_trocr.upper().replace('I', '1').replace('O', '0').replace('Q', '0')
                    
                    # Store results
                    self.ocr_results[track_id] = {
                        'easyocr': license_num_easy,
                        'trocr': license_num_trocr,
                        'timestamp': timestamp,
                        'bbox': (x1, y1, x2, y2)
                    }
                    
                    self.ocr_queue.task_done()
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"OCR worker error: {e}")
                continue

    def detection(self, frame):
        self.frame_count += 1
        self.fps_counter += 1
        
        # Skip frames for performance
        if self.frame_count % self.frame_skip != 0:
            # Still process OCR results and display existing detections
            if self.type == "hongkong":
                self._process_ocr_results(frame)
            # Show FPS
            self._display_fps(frame)
            cv2.imshow("RTSP License Detection", frame)
            return frame
        
        current_time = time.time()
        results = self.detector.track(frame, persist=True, conf=0.6)
        
        if results[0].boxes and results[0].boxes.id is not None:
            xyxys = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for xyxy, track_id in zip(xyxys, track_ids):
                if track_id == np.int64(9):
                    continue
                    
                x1, y1, x2, y2 = map(int, xyxy)
                
                if self.type == "hongkong":
                    # Queue for async OCR processing if not already processed
                    if track_id not in self.id_final_easyocr and track_id not in self.ocr_results:
                        if not self.ocr_queue.full():
                            license_image = frame[y1:y2, x1:x2].copy()
                            self.ocr_queue.put((track_id, license_image, current_time, x1, y1, x2, y2))
                    
                    # Check for completed OCR results
                    if track_id in self.ocr_results:
                        self._handle_ocr_result(track_id, frame, current_time)
                    
                    # Display if finalized
                    if track_id in self.id_final_easyocr:
                        frame = self.visualization(frame, self.id_final_easyocr[track_id], x1, y1, x2, y2)
                        
                elif self.type == "china":
                    frame = self.reader_logging_net(track_id, frame, x1, y1, x2, y2, current_time)
        
        # Show FPS
        self._display_fps(frame)
        # Display realtime output
        cv2.imshow("RTSP License Detection", frame)
        return frame
    
    def _process_ocr_results(self, frame):
        """Process completed OCR results on skipped frames"""
        for track_id in list(self.ocr_results.keys()):
            if track_id not in self.id_final_easyocr:
                self._handle_ocr_result(track_id, frame, time.time())
    
    def _handle_ocr_result(self, track_id, frame, current_time):
        """Handle completed OCR results"""
        result = self.ocr_results[track_id]
        license_easy = result['easyocr']
        license_trocr = result['trocr']
        
        if license_easy and license_easy != "":
            if track_id not in self.id_dict_easyocr:
                self.id_dict_easyocr[track_id] = {}
            if license_easy not in self.id_dict_easyocr[track_id]:
                self.id_dict_easyocr[track_id][license_easy] = 0
            self.id_dict_easyocr[track_id][license_easy] += 1
            if self.id_dict_easyocr[track_id][license_easy] > self.FRAME_THRESHOLD:
                self.id_final_easyocr[track_id] = license_easy
        
        if license_trocr and license_trocr != "":
            if track_id not in self.id_dict_trocr:
                self.id_dict_trocr[track_id] = {}
            if license_trocr not in self.id_dict_trocr[track_id]:
                self.id_dict_trocr[track_id][license_trocr] = 0
            self.id_dict_trocr[track_id][license_trocr] += 1
            if self.id_dict_trocr[track_id][license_trocr] > self.FRAME_THRESHOLD:
                self.id_final_trocr[track_id] = license_trocr
        
        # Save image and write to CSV when both are finalized
        if (track_id in self.id_final_easyocr and track_id in self.id_final_trocr and 
            track_id not in self.written_track_ids):
            self.csv_writer.writerow([track_id, f"{self.output_dir}/easyocr_{track_id}_{self.id_final_easyocr[track_id]}_{int(current_time)}.jpg",
                                    self.id_final_easyocr[track_id], self.id_final_trocr[track_id], current_time])
            self.written_track_ids.add(track_id)
        
        # Clean up processed results
        del self.ocr_results[track_id]
    
    def _display_fps(self, frame):
        """Display FPS counter on frame"""
        current_time = time.time()
        if current_time - self.fps_start_time >= 1.0:
            fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
            self.current_fps = fps
        
        if hasattr(self, 'current_fps'):
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
                    # Try to reconnect
                    self.video_cap.release()
                    time.sleep(1)
                    self.video_cap = cv2.VideoCapture(self.rtsp_url)
                    self.video_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.video_cap.set(cv2.CAP_PROP_FPS, 15)
                    self.video_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    self.video_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    if not self.video_cap.isOpened():
                        print("Failed to reconnect to RTSP stream")
                        break
        except KeyboardInterrupt:
            print("\nStopping RTSP stream processing...")
        finally:
            # Cleanup
            if hasattr(self, 'ocr_thread_running'):
                self.ocr_thread_running = False
                self.ocr_queue.put(None)  # Signal thread to stop
                if hasattr(self, 'ocr_thread'):
                    self.ocr_thread.join(timeout=2)
            
            self.video_cap.release()
            self.csv_file.close()
            cv2.destroyAllWindows()
            print(f"License detections saved to {self.csv_file_path}")


if __name__ == "__main__":
    rtsp_url="rtsp://admin:rsxx1111@192.168.10.28/1"
    # rtsp_url="rtsp://localhost:8554/"
    try:
        # Choose license type: "hongkong" or "china"
        detector = rtsp_realtime_detection(rtsp_url, license_type="hongkong")
        detector.main()
    except Exception as e:
        print(f"Error processing RTSP stream {rtsp_url}: {e}")