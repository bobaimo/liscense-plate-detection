#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LPRNet Validation Script
Validates LPRNet model predictions against ground truth from image filenames.
Author: Claude
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LPRNet_reader import LPRNetInference


def validate_lprnet(sample_dir='./sample/0/', model_path='./models/LPRNet_model/weights/Final_LPRNet_model.pth', yolo_model_path='./models/license_plate.pt'):
    """
    Validate LPRNet model against test images with YOLO license plate detection
    
    Args:
        sample_dir: Directory containing test images
        model_path: Path to trained model weights
        yolo_model_path: Path to YOLO license plate detection model
    """
    
    # Initialize the YOLO model
    print(f"Loading YOLO model from: {yolo_model_path}")
    try:
        yolo_model = YOLO(yolo_model_path)
        print("YOLO model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    # Initialize the LPRNet inference model
    print(f"Loading LPRNet model from: {model_path}")
    try:
        lpr_inference = LPRNetInference(model_path=model_path)
        print("LPRNet model loaded successfully!")
    except Exception as e:
        print(f"Error loading LPRNet model: {e}")
        return
    
    # Get all image files in the sample directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(sample_dir, ext)))
    
    if not image_files:
        print(f"No image files found in {sample_dir}")
        return
    
    print(f"Found {len(image_files)} test images")
    
    # Validation results
    correct_predictions = 0
    total_predictions = 0
    no_detection_count = 0
    results = []
    
    print("\nValidation Results:")
    print("-" * 90)
    print(f"{'Image':<25} {'Ground Truth':<15} {'Prediction':<15} {'Detections':<8} {'Match':<8}")
    print("-" * 90)
    
    for image_path in sorted(image_files):
        # Extract ground truth from filename (stem without extension)
        filename = os.path.basename(image_path)
        ground_truth = os.path.splitext(filename)[0]
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Run YOLO detection
            results_yolo = yolo_model(image)
            
            # Extract license plate detections
            detections = []
            for result in results_yolo:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        
                        # Filter by confidence threshold
                        if conf > 0.5:
                            detections.append((x1, y1, x2, y2, conf))
            
            prediction = "NO_DETECTION"
            num_detections = len(detections)
            
            if detections:
                # Use the detection with highest confidence
                best_detection = max(detections, key=lambda x: x[4])
                x1, y1, x2, y2, conf = best_detection
                
                # Crop the license plate region
                cropped_plate = image[y1:y2, x1:x2]
                
                if cropped_plate.size > 0:
                    # Make prediction on cropped plate
                    prediction = lpr_inference.predict(cropped_plate)
                else:
                    prediction = "CROP_ERROR"
            else:
                no_detection_count += 1
            
            # Check if prediction matches ground truth
            is_correct = prediction == ground_truth
            if is_correct:
                correct_predictions += 1
            
            total_predictions += 1
            
            # Store result
            results.append({
                'image': filename,
                'ground_truth': ground_truth,
                'prediction': prediction,
                'num_detections': num_detections,
                'correct': is_correct
            })
            
            # Print result
            match_symbol = "✓" if is_correct else "✗"
            print(f"{filename:<25} {ground_truth:<15} {prediction:<15} {num_detections:<8} {match_symbol:<8}")
            
        except Exception as e:
            print(f"{filename:<25} {ground_truth:<15} {'ERROR':<15} {'0':<8} {'✗':<8}")
            print(f"  Error: {e}")
            total_predictions += 1
    
    print("-" * 90)
    
    # Calculate and display statistics
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\nValidation Summary:")
        print(f"Total images: {total_predictions}")
        print(f"Images with no detection: {no_detection_count}")
        print(f"Images with detection: {total_predictions - no_detection_count}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Incorrect predictions: {total_predictions - correct_predictions}")
        print(f"Overall accuracy: {accuracy:.2f}%")
        
        # Calculate accuracy for images with detections
        if total_predictions - no_detection_count > 0:
            detection_accuracy = (correct_predictions / (total_predictions - no_detection_count)) * 100
            print(f"Accuracy on detected plates: {detection_accuracy:.2f}%")
        
        # Show some error analysis
        incorrect_results = [r for r in results if not r['correct']]
        if incorrect_results:
            print(f"\nError Analysis ({len(incorrect_results)} errors):")
            for result in incorrect_results[:10]:  # Show first 10 errors
                print(f"  {result['image']}: '{result['ground_truth']}' → '{result['prediction']}'")
            if len(incorrect_results) > 10:
                print(f"  ... and {len(incorrect_results) - 10} more errors")
    else:
        print("No predictions were made.")


def main():
    parser = argparse.ArgumentParser(description='Validate LPRNet model performance')
    parser.add_argument('--sample_dir', default='./sample/0/', 
                       help='Directory containing test images')
    parser.add_argument('--model', default='./models/LPRNet_model/weights/Final_LPRNet_model.pth',
                       help='Path to trained model weights')
    parser.add_argument('--yolo_model', default='./models/license_plate.pt',
                       help='Path to YOLO license plate detection model')
    
    args = parser.parse_args()
    
    # Check if sample directory exists
    if not os.path.exists(args.sample_dir):
        print(f"Error: Sample directory '{args.sample_dir}' does not exist")
        return
    
    # Check if model files exist
    if not os.path.exists(args.model):
        print(f"Error: LPRNet model file '{args.model}' does not exist")
        return
    
    if not os.path.exists(args.yolo_model):
        print(f"Error: YOLO model file '{args.yolo_model}' does not exist")
        return
    
    # Run validation
    validate_lprnet(args.sample_dir, args.model, args.yolo_model)


if __name__ == "__main__":
    main()