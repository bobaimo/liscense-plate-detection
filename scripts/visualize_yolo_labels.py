import cv2
import os
import sys
from pathlib import Path

class YOLOVisualizer:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        self.height, self.width = self.image.shape[:2]
        self.class_names = ["license_plate"]
        
    def yolo_to_bbox(self, center_x, center_y, width, height):
        """Convert YOLO format to bounding box coordinates"""
        x1 = int((center_x - width/2) * self.width)
        y1 = int((center_y - height/2) * self.height)
        x2 = int((center_x + width/2) * self.width)
        y2 = int((center_y + height/2) * self.height)
        return x1, y1, x2, y2
    
    def load_labels(self):
        """Load YOLO labels from the label directory"""
        image_path = Path(self.image_path)
        image_dir = image_path.parent
        video_dir = image_dir.parent
        label_dir = video_dir / "label"
        label_path = label_dir / f"{image_path.stem}.txt"
        
        boxes = []
        
        if not label_path.exists():
            print(f"No label file found: {label_path}")
            return boxes
            
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    parts = line.split()
                    if len(parts) != 5:
                        print(f"Warning: Invalid format at line {line_num}: {line}")
                        continue
                        
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Validate YOLO coordinates (should be between 0 and 1)
                    if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        print(f"Warning: Invalid YOLO coordinates at line {line_num}: {line}")
                        continue
                    
                    x1, y1, x2, y2 = self.yolo_to_bbox(center_x, center_y, width, height)
                    
                    boxes.append({
                        'class_id': class_id,
                        'bbox': (x1, y1, x2, y2),
                        'yolo': (center_x, center_y, width, height)
                    })
                    
                except ValueError as e:
                    print(f"Warning: Could not parse line {line_num}: {line} - {e}")
                    continue
        
        print(f"Loaded {len(boxes)} bounding boxes from {label_path}")
        return boxes
    
    def draw_boxes(self, boxes):
        """Draw bounding boxes on the image"""
        image_with_boxes = self.image.copy()
        
        for i, box in enumerate(boxes):
            class_id = box['class_id']
            x1, y1, x2, y2 = box['bbox']
            center_x, center_y, width, height = box['yolo']
            
            # Green color for bounding boxes
            color = (0, 255, 0)
            thickness = 2
            
            # Draw rectangle
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)
            
            # Add label
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
            label = f"{class_name} ({i+1})"
            
            # Calculate label background size
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Draw label background
            cv2.rectangle(image_with_boxes, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0] + 10, y1), color, -1)
            
            # Draw label text
            cv2.putText(image_with_boxes, label, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Print YOLO coordinates
            print(f"Box {i+1}: class={class_id}, YOLO=({center_x:.6f}, {center_y:.6f}, {width:.6f}, {height:.6f})")
            print(f"         Pixel coords=({x1}, {y1}, {x2}, {y2})")
        
        return image_with_boxes
    
    def visualize(self):
        """Main visualization function"""
        print(f"Loading image: {self.image_path}")
        print(f"Image dimensions: {self.width}x{self.height}")
        
        # Load labels
        boxes = self.load_labels()
        
        if not boxes:
            print("No bounding boxes found. Displaying original image.")
            image_with_boxes = self.image.copy()
            # Add text indicating no labels
            cv2.putText(image_with_boxes, "No labels found", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # Draw boxes on image
            image_with_boxes = self.draw_boxes(boxes)
            
            # Add summary text
            cv2.putText(image_with_boxes, f"Boxes: {len(boxes)}", 
                       (self.width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display image
        cv2.namedWindow('YOLO Label Visualization', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLO Label Visualization', image_with_boxes)
        
        print("\nControls:")
        print("- Press any key to close the window")
        print("- Press 's' to save the visualization")
        
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('s'):
            # Save visualization
            output_path = f"{Path(self.image_path).stem}_visualization.jpg"
            cv2.imwrite(output_path, image_with_boxes)
            print(f"Visualization saved: {output_path}")
        
        cv2.destroyAllWindows()

def main():
    if len(sys.argv) != 2:
        print("Usage: python visualize_yolo_labels.py <image_path>")
        print("Example: python visualize_yolo_labels.py images/frame001.jpg")
        return
    
    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    try:
        visualizer = YOLOVisualizer(image_path)
        visualizer.visualize()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()