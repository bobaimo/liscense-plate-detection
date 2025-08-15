import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import cv2
import numpy as np

class visual_llm:
     
    def __init__(self):
        self.model_path="Qwen/Qwen2-VL-7B-Instruct"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path, 
                                                                    torch_dtype=torch.bfloat16, 
                                                                    attn_implementation="sdpa",
                                                                    device_map="auto"
                                                                    )
        self.processor = AutoProcessor.from_pretrained(self.model_path)

    def prompt_detection(self,image,prompt):
        # Handle cv2 image conversion if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB and then to PIL Image
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_image)
        elif isinstance(image, str):
            # Handle file path
            image = Image.open(image).convert('RGB')
        # If already PIL Image, use as is
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text],
                                images=image_inputs,
                                videos=video_inputs,
                                padding=True,
                                return_tensors="pt",
                                )
        inputs = inputs.to("cuda")
        
        # Generate response
        generated_ids = self.model.generate(**inputs, max_new_tokens=10)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
                
        # Check if answer is "no" and copy to invalid directory

        if answer.lower().strip() in ['yes', 'yes.']:
            return True
        else:
            return False
        

if __name__=="__main__":
    vlm=visual_llm()
    image= Image.open("plate/5884210512.jpg").convert('RGB')
    print(vlm.prompt_detection(image,"Is the image a car license plate? Answer only 'yes' or 'no'."))


