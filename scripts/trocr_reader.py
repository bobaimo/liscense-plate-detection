from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
from easyocr_reader import easyocr_reader



class trocr_reader:

    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-printed", use_fast=True)
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-printed")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def __str__(self):
        return "trocr"

    def ocr_with_trocr(self,image):

        # Process image
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.to(self.device)

        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        # Decode text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return generated_text
    
    def split_image(self,image,bbox_list):
        image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        target_width,target_height=self.get_size(image)
        resized = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        image_list=[]
        for bbox in bbox_list:
            x1,x2,y1,y2=self.corner_to_xyxy(bbox)
            image_list.append(resized[y1:y2,x1:x2])
        if len(image_list)==0:
            return [resized]
        else:
            return image_list


    def get_size(self,image):
        height, width = image.shape[:2]
        ratio = width / height if height > 0 else 0
        if ratio<3: 
            scale_factor = 340/width
        else:
            scale_factor = 450/width
        return int(scale_factor*width), int(scale_factor*height)
    
    def read_plate(self,image,bbox_list):
        image_list=self.split_image(image,bbox_list)
        license_num=""
        for i in range (len(image_list)):
            processing_image=image_list[i]
            # cv2.imshow('Image 1', processing_image)
            # cv2.waitKey(2000)
            if processing_image.size !=0:
                license_num+=self.ocr_with_trocr(processing_image).replace(" ", "")
        return license_num,bbox_list
    
    def image_visualization(self,image,bbox):
        image_list=self.split_image(image,bbox)
        if len(image_list)>1:
            cv2.imshow('Image 1', image_list[0])
            cv2.imshow('Image 2', image_list[1])
        else:
            cv2.imshow('Image 1', image_list[0])
        cv2.waitKey(2000)
            
    def corner_to_xyxy(self,bbox):
        x1=int(bbox[0][0])
        x2=int(bbox[1][0])
        y1=int(bbox[0][1])
        y2=int(bbox[2][1])
        return x1,x2,y1,y2
    
if __name__=="__main__":
    # image= cv2.imread("dataset/carpark_1/bounding_box/frame_000675_1.jpg")
    # image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # for i in range (900,946,15):
    #     image= Image.open("dataset/carpark_1/bounding_box/frame_000"+str(i)+"_1.jpg")
    #     result = ocr_with_trocr(image)

    #     print(f"OCR Result: {result}")

    t=trocr_reader()
    r=easyocr_reader()
    for directory in ["","_original"]:
        print(f"directory:{directory}")
        for i in range (1305,1321,15):
            image= cv2.imread(f"dataset/carpark_1{directory}/bounding_box/frame_00{i}_1.jpg")
            license_num,bbox=r.read_plate(image,[])
            print(f"easyocr:{license_num}")
            t.image_visualization(image,bbox)
            license_num,_=r.read_plate(image,bbox)
            print(f"trocr:{license_num}")

    
        