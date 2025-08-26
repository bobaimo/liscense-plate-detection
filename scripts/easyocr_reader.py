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

    def read_plate(self,image,bbox_list):
        processed_image=self.image_processing(image)
        result=self.reader.readtext(processed_image)
        text_list=[]
        for i in range (len(result)):
            bbox_list.append(result[i][0])
            text_list.append(result[i][1])
        license_num="".join(text_list).replace(" ", "")
        # license_num = "".join(self.reader.readtext(image,detail=0)).replace(" ", "")
        return license_num,bbox_list
    
    def get_size(self,image):
        height, width = image.shape[:2]
        ratio = width / height if height > 0 else 0
        if ratio<3: 
            scale_factor = 340/width
        else:
            scale_factor = 450/width
        return int(scale_factor*width), int(scale_factor*height)
           
if __name__=="__main__":
    r=easyocr_reader()
    # for i in range (900,946,15):
    #     image= Image.open("dataset/carpark_1/bounding_box/frame_000"+str(i)+"_1.jpg")
    #     # r.image_visualization(r.image_processing(image))
    #     print(r.read_plate(image))
    for directory in ["","_original"]:
        print(f"directory:{directory}")
        for i in range (1305,1321,15):
            image= cv2.imread(f"dataset/carpark_1{directory}/bounding_box/frame_00{i}_1.jpg")
            license_num,bbox=r.read_plate(image,[])
            print(license_num,bbox)

