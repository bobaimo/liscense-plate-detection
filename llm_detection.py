from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from pathlib import Path
import shutil

model_path = "Qwen/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="sdpa",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(model_path)

# Directory containing bounding box images
boundingbox_dir = Path("dataset/hkstp_high/boundingbox")
invalid_dir = Path("dataset/hkstp_high/invalid")
valid_dir = Path("dataset/hkstp_high/valid")

# Create directories if they don't exist
invalid_dir.mkdir(parents=True, exist_ok=True)
valid_dir.mkdir(parents=True, exist_ok=True)

# Check if directory exists
if not boundingbox_dir.exists():
    print(f"Directory {boundingbox_dir} does not exist!")
    exit(1)

# Get all image files in the directory
image_files = [f for f in boundingbox_dir.glob("*.jpg") if f.is_file()]

if not image_files:
    print(f"No .jpg files found in {boundingbox_dir}")
    exit(1)

print(f"Found {len(image_files)} images to process")

# Process each image
invalid_count = 0
valid_count = 0
for image_path in sorted(image_files):
    try:
        image = Image.open(image_path).convert('RGB')
        
        # Question to ask about each image
        question = "Is the image a car license plate? Answer only 'yes' or 'no'."
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # Prepare the inputs
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Generate response
        generated_ids = model.generate(**inputs, max_new_tokens=10)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        answer = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"{image_path.name}: {answer}")
        
        # Check if answer is "no" and copy to invalid directory
        if answer.lower().strip() in ['no', 'no.']:
            destination = invalid_dir / image_path.name
            shutil.copy2(str(image_path), str(destination))
            print(f"  -> Copied {image_path.name} to invalid directory")
            invalid_count += 1
        # Check if answer is "yes" and copy to valid directory
        elif answer.lower().strip() in ['yes', 'yes.']:
            # # Follow-up question about black background
            # follow_up_question = "Are the letters white in color? Answer only 'yes' or 'no'."
            
            # follow_up_messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {
            #                 "type": "image",
            #                 "image": image,
            #             },
            #             {"type": "text", "text": follow_up_question},
            #         ],
            #     }
            # ]
            
            # # Prepare the follow-up inputs
            # follow_up_text = processor.apply_chat_template(follow_up_messages, tokenize=False, add_generation_prompt=True)
            # follow_up_image_inputs, follow_up_video_inputs = process_vision_info(follow_up_messages)
            # follow_up_inputs = processor(
            #     text=[follow_up_text],
            #     images=follow_up_image_inputs,
            #     videos=follow_up_video_inputs,
            #     padding=True,
            #     return_tensors="pt",
            # )
            # follow_up_inputs = follow_up_inputs.to("cuda")
            
            # # Generate follow-up response
            # follow_up_generated_ids = model.generate(**follow_up_inputs, max_new_tokens=10)
            # follow_up_generated_ids_trimmed = [
            #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(follow_up_inputs.input_ids, follow_up_generated_ids)
            # ]
            # follow_up_answer = processor.batch_decode(
            #     follow_up_generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            # )[0]
            
            # if follow_up_answer.lower().strip() in ['yes', 'yes.']:
            #     destination = invalid_dir / image_path.name
            #     shutil.copy2(str(image_path), str(destination))
            #     print(f"  -> Copied {image_path.name} to invalid directory (white letter)")
            #     invalid_count += 1
            # elif follow_up_answer.lower().strip() in ['no', 'no.']:
                # destination = valid_dir / image_path.name
                # shutil.copy2(str(image_path), str(destination))
                # print(f"  -> Copied {image_path.name} to valid directory")
                # valid_count += 1
            destination = valid_dir / image_path.name
            shutil.copy2(str(image_path), str(destination))
            print(f"  -> Copied {image_path.name} to valid directory")
            valid_count += 1
        
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")

print(f"Processing complete! Copied {valid_count} images to valid directory and {invalid_count} images to invalid directory.")