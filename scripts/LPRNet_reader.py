#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LPRNet Inference Script
License Plate Recognition inference using pretrained model.
Author: Claude
"""

import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.LPRNet_model.data.load_data import CHARS
from models.LPRNet_model.LPRNet import build_lprnet


class LPRNetInference:
    def __init__(self, model_path='./models/LPRNet_model/weights/Final_LPRNet_model.pth', 
                 img_size=[94, 24], lpr_max_len=8, device='auto'):
        """
        Initialize LPRNet inference model
        
        Args:
            model_path: Path to pretrained model weights
            img_size: Input image size [width, height]
            lpr_max_len: Maximum license plate length
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.img_size = img_size
        self.lpr_max_len = lpr_max_len
        
        # Set device
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Build model
        self.model = build_lprnet(
            lpr_max_len=lpr_max_len, 
            phase=False,  # inference mode
            class_num=len(CHARS), 
            dropout_rate=0
        )
        
        # Load pretrained weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pretrained model from {model_path}")
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
    def __str__(self):
        return "LPRNet"
    
    def preprocess_image(self, image_input):
        """
        Preprocess input image for model inference
        
        Args:
            image_input: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            Preprocessed image tensor
        """
        # Handle different input types
        if isinstance(image_input, str):
            # File path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            img = cv2.imread(image_input)
        elif isinstance(image_input, Image.Image):
            # PIL Image
            img = cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            # Numpy array
            img = image_input.copy()
        else:
            raise ValueError("Unsupported image input type")
        
        if img is None:
            raise ValueError("Failed to load image")
        
        # Resize image
        height, width = img.shape[:2]
        if height != self.img_size[1] or width != self.img_size[0]:
            img = cv2.resize(img, tuple(self.img_size))
        
        # Normalize
        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        
        # Convert to tensor format (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # Add batch dimension
        
        return img_tensor
    
    def decode_prediction(self, prediction):
        """
        Decode model prediction to license plate string
        
        Args:
            prediction: Model output tensor
            
        Returns:
            Decoded license plate string
        """
        prediction = prediction.cpu().detach().numpy()
        
        # Greedy decode
        preb = prediction[0, :, :]  # Remove batch dimension
        preb_label = []
        
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        
        # Remove repeated characters and blank labels
        no_repeat_blank_label = []
        pre_c = preb_label[0]
        
        if pre_c != len(CHARS) - 1:  # Not blank
            no_repeat_blank_label.append(pre_c)
            
        for c in preb_label:
            if (pre_c == c) or (c == len(CHARS) - 1):  # Same as previous or blank
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        
        # Convert indices to characters
        result = ""
        for idx in no_repeat_blank_label:
            result += CHARS[idx]
            
        return result
    
    def predict(self, image_input):
        """
        Predict license plate from image
        
        Args:
            image_input: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            Predicted license plate string
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image_input)
        img_tensor = img_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            prediction = self.model(img_tensor)
        
        # Decode prediction
        result = self.decode_prediction(prediction)
        
        return result


def main():
    parser = argparse.ArgumentParser(description='LPRNet Inference')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='./models/LPRNet_model/weights/Final_LPRNet_model.pth', 
                       help='Path to pretrained model')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Initialize inference model
    lpr_inference = LPRNetInference(
        model_path=args.model,
        device=args.device
    )
    
    # Predict
    try:
        result = lpr_inference.predict(args.image)
        print(f"Predicted license plate: {result}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()