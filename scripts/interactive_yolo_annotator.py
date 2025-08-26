import cv2
import os
from pathlib import Path
import sys

class YOLOAnnotator:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        self.original_image = self.image.copy()
        self.height, self.width = self.image.shape[:2]
        
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_box = None
        self.boxes = []
        self.current_class = 0
        self.class_names = ["license_plate"]
        
        cv2.namedWindow('YOLO Annotator', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('YOLO Annotator', self.mouse_callback)
        
        print("Controls:")
        print("- Click and drag to draw bounding box")
        print("- Press 'r' to reset current drawing")
        print("- Press 'u' to undo last box")
        print("- Press 's' to save annotations")
        print("- Press 'c' to clear all boxes")
        print("- Press 'q' to quit")
    
    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.current_box = (self.start_point[0], self.start_point[1], 
                                 self.end_point[0], self.end_point[1])
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                
                # Ensure x1,y1 is top-left and x2,y2 is bottom-right
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                
                if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:  # Minimum box size
                    self.boxes.append({
                        'class': self.current_class,
                        'bbox': (x1, y1, x2, y2)
                    })
                    print(f"Added box {len(self.boxes)}: class={self.current_class}, "
                          f"bbox=({x1},{y1},{x2},{y2})")
                    print(f"YOLO format: {self.bbox_to_yolo(x1, y1, x2, y2)}")
                
                self.current_box = None
                self.start_point = None
                self.end_point = None
    
    def bbox_to_yolo(self, x1, y1, x2, y2):
        """Convert bounding box to YOLO format (normalized)"""
        center_x = (x1 + x2) / 2.0 / self.width
        center_y = (y1 + y2) / 2.0 / self.height
        box_width = (x2 - x1) / self.width
        box_height = (y2 - y1) / self.height
        return f"{self.current_class} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
    
    
    def draw_boxes(self):
        """Draw all bounding boxes on the image"""
        self.image = self.original_image.copy()
        
        # Draw saved boxes
        for i, box_info in enumerate(self.boxes):
            class_id = box_info['class']
            x1, y1, x2, y2 = box_info['bbox']
            
            # Green color for single class
            color = (0, 255, 0)
            
            cv2.rectangle(self.image, (x1, y1), (x2, y2), color, 2)
            
            # Add class label
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            label = f"{class_name} ({i+1})"
            cv2.putText(self.image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw current box being drawn
        if self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(self.image, f"Drawing... Class: {self.current_class}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        
        # Show box count
        cv2.putText(self.image, f"Boxes: {len(self.boxes)}", 
                   (self.width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def save_annotations(self):
        """Save annotations in YOLO format"""
        image_path = Path(self.image_path)
        image_dir = image_path.parent
        video_dir = image_dir.parent
        label_dir = video_dir / "label"
        
        # Create label directory if it doesn't exist
        label_dir.mkdir(exist_ok=True)
        
        # Save YOLO format (.txt)
        yolo_path = label_dir / f"{image_path.stem}.txt"
        with open(yolo_path, 'w') as f:
            for box_info in self.boxes:
                x1, y1, x2, y2 = box_info['bbox']
                yolo_line = self.bbox_to_yolo(x1, y1, x2, y2)
                f.write(yolo_line + '\n')
        
        print(f"Annotations saved: {yolo_path}")
    
    
    def run(self):
        """Main annotation loop"""
        
        while True:
            self.draw_boxes()
            cv2.imshow('YOLO Annotator', self.image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_annotations()
            elif key == ord('r'):
                # Reset current drawing
                self.drawing = False
                self.current_box = None
                self.start_point = None
                self.end_point = None
            elif key == ord('u'):
                # Undo last box
                if self.boxes:
                    removed = self.boxes.pop()
                    print(f"Removed box: {removed}")
            elif key == ord('c'):
                # Clear all boxes
                self.boxes = []
                print("Cleared all boxes")
        
        cv2.destroyAllWindows()

def main():
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    annotator = YOLOAnnotator(image_path)
    annotator.run()

if __name__ == "__main__":
    main()