import easyocr
import cv2
from PIL import Image

class easyocr_reader:

    def __init__(self):
        self.reader=easyocr.Reader(lang_list=['en'],gpu=True)
    
    def __str__(self):
        return "easyocr"

    def image_processing(self,image):
        # image_rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # target_width,target_height=self.get_size(image)
        # resized = cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        # return resized

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target_width,target_height=self.get_size(image)
        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(resized, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
        enhanced = clahe.apply(blurred)
        return enhanced

    def image_visualization(self,image,bbox_list=[]):
        cv2.imshow("",image)
        cv2.waitKey(2000)

    def read_plate(self,image,bbox_list=[]):
        processed_image=self.image_processing(image)
        result=self.reader.readtext(processed_image)
        text_list=[]
        for i in range (len(result)):
            bbox_corner=result[i][0]
            bbox_list.append(self.corner_to_xyxy(bbox_corner))
            text_list.append(result[i][1])
        license_num="".join(text_list).replace(" ", "")
        return license_num,bbox_list
    
    def get_size(self,image):
        height, width = image.shape[:2]
        ratio = width / height if height > 0 else 0
        if ratio<3: 
            scale_factor = 340/width
        else:
            scale_factor = 450/width
        return int(scale_factor*width), int(scale_factor*height)
    
    def corner_to_xyxy(self,bbox):
        x1=int(bbox[0][0])
        x2=int(bbox[1][0])
        y1=int(bbox[0][1])
        y2=int(bbox[2][1])
        return x1,x2,y1,y2
           
if __name__=="__main__":
    r=easyocr_reader()
    for directory in ["","_original"]:
        print(f"directory:{directory}")
        for i in range (1305,1321,15):
            image= cv2.imread(f"dataset/carpark_1{directory}/bounding_box/frame_00{i}_1.jpg")
            license_num,bbox=r.read_plate(image)
            print(license_num,bbox)

