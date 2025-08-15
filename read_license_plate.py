import easyocr
import cv2

class ocr_reader:

    def __init__(self):
        self.reader=easyocr.Reader(['en'])

    def image_processing(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target_width,target_height=self.get_size(image)
        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(10, 10))
        enhanced = clahe.apply(blurred)
        return enhanced

    def image_visualization(self,image):
        cv2.imshow("",image)
        cv2.waitKey(2000)

    def read_plate(self,image):
        processed_image=self.image_processing(image)
        license_num = "".join(self.reader.readtext(processed_image,detail=0))
        return license_num
    
    def get_size(self,image):
        height, width = image.shape[:2]
        ratio = width / height if height > 0 else 0
        print(f"width: {width}, height: {height}, ratio: {ratio:.2f}")
        if ratio<3: 
            scale_factor = 340/width
        else:
            scale_factor = 450/width
        return int(scale_factor*width), int(scale_factor*height)
           
if __name__=="__main__":
    # list=["2","4","14","24","43","45"]
    # list=["39","40","52","56"]
    list=["1","2","3","5","24","25","26","29","31"]
    scale_factor=2
    r=ocr_reader()
    for int in list:
        image = cv2.imread('cropped_box/carpark_2/track_'+int+'_with_box.jpg')
        r.image_visualization(r.image_processing(image))
        print(r.read_plate(image))