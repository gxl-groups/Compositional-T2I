from openai import OpenAI
import random
import re
import json
import argparse
import os

my_key = "your api key"

def gpt2layout(caption):

    client = OpenAI(
        api_key=my_key,
        base_url= "https://api.openai.com/v1"
    )
    gpt_messages = [{
        "role": "system",
        "content": """You are a virtual python interpreter. For an input "caption", you need to execute the following 
        python pseudocode. For an undefined function, you need to use your thinking and reasoning skills to give an output.
        The caption is a description of a picture, the code's final goal is to get each object's layout which appear in the picture.
        # CODE START
        def get_number(obj, caption):
            # From the quantifiers in the caption, figure out the quantity of the object. 
            # If it is some abstract, non-concrete quantifier, you need to determine an explicit quantity for it.
            
            
        def get_location_description(obj, caption):
            # Based on the caption, analyze the spatial relationship of the object in the picture
            # If it's not explicitly stated, you have figure out what's most possible situation. You must describe your analysis, not simply the results.
            # Then determine which area of the picture is to be located, such as the left, right, top left and so on，the more specific the better. 
            
        
        def get_layout(obj, num, location_description):
             # Based on the quantity and description, reason or guess a reasonable layout in the picture for the "object-id". the layout should be (x1, y1, x2, y2), where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate.Note that he frame should be 256*256, don't exceed.
             # Each object should have its own unique layout that doesn't overshadow other layouts.
             
        
        objects_list = parse_objects(caption)
        objects_attribute_dict = {}  # Create an empty attribute dictionary
        for obj in objects_list:
            # here you get the number and location description for each object, you need to infer from the caption
            number = get_number(obj, caption)  
            location_description = get_location_description(obj, caption)
            # Create an attribute dictionary for the object
            obj_attributes = {"number": number, "location description": location_description}
            
            # Add the object and its attributes to the overall dictionary
            objects_attribute_dict[obj] = obj_attributes
        print("Attribute Dict:", objects_attribute_dict)  
          
        objects_layout_dict = {}
        for obj in objects_attribute_dict:
            num = objects_attribute_dict[obj]["number"]
            location_description = objects_attribute_dict[obj]["location"]
            for id in range(num):
                layout = get_layout(obj, num, location_description)
                objects_layout_dict[f"{obj}-{id}"] = layout
        print("Layout Dict:", objects_layout_dict)
        ### CODE END
        Here are some example:
        ### Example1:
        Input caption: a white dog on the left of two cats.
        Answer:
        Attribute Dict: {"white dog": {"number": 1, "location description": "Its location relationship is mentioned, it's on the left of the two cats. So I think it is on the left side of the picture."}
        "cats": {"number": 2, "location description": "There are a dog on the left of they mentioned in the caption mentioned, so i determined that the two cats should on the right side of the picture, and one is on the top right and one is on the bottom right"}}
        Layout Dict: {"white dog-0": (25, 30, 103, 208), "cats-0": (111, 29, 230, 113), "cats-1": (120, 140, 241, 250))}
        ### Example2:
        Input caption: Two person and a horse.
        Answer:
        Attribute Dict: {"person": {"number": 2, "location description": "No specific location mentioned, so I assume that one person are ridding the horse and another stood by, leading the horse. So one person is on the horse, the other person is in front of the horse, and I'm assuming the horse is to the right, so the other person is on the right of the horse. And The specific position of the two person in the frame depends on the position of the horse."},
         "horse":{"number":1, "location description": "No specific location mentioned, according to the previous guess of person's position, the horse should be the main body of the picture, so I think the horse is in the center of the picture"}}
        Layout Dict: {"person-0": (95, 18, 150, 18), "person-1": (180, 125, 238, 231), "horse-0": (80, 135, 170, 208)}
        ### Please use your imagination to reply to my next query. Pay attention that Never Output The Same Layout As Other Objects!!!
        """.strip()
    }, {"role": "user",
        "content": f"""Input caption: {caption}"""}
    ]

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=gpt_messages,
        temperature=0.75,
    )
    print(f"response:\n{response.choices[0].message.content}")

    match = re.search(r'\{([^{}]*)\}.*$', response.choices[0].message.content, re.IGNORECASE)
    if match:
        json_part = "{" + match.group(1) + "}"
        parsed_dict = json.loads(json_part.replace("(", "[").replace(")", "]"))
        return parsed_dict
    else:
        print("no JSON data matched")
        return {"none": (0, 0, 255, 255)}


'''
, {
        "role": "user",
        "content": "Input caption: a dog and two birds."
    }
    , {
        "role": "assistant",
        "content": """
        Before i simulate the code, I repeat the rule:
        1. When i simulate get_position(), if the caption doesn't give an exact location, I will first guess a relative position of the object to other objects, and then infer where the object is in the frame.
        2. When i simulate get_layout(), i will note that never repeat the layout.
        Attribute Dict: {"dog": {"number": 1, "location_description": "No specific location mentioned, so i can guess it's on the left of two birds, so it should be on the left side of the picture"}
        "birds": {"number": 2, "location_description": "No specific location mentioned, but i guessed the dog is on the left of two birds before, so the two birds should on the right side of the picture"}}
        Layout Dict: {"white dog-0": (30, 40, 80, 223), "cats-0": (92, 29, 230, 113), "cats-1": (101, 140, 241, 250))}
        """.strip()
    }
'''
'''
 caption: A pink bathroom with a mirror and a green sink.
response:
Attribute Dict: {"pink bathroom": {"number": 1, "location_description": "It's the main focus of the picture, so it should be in the center"},
        "mirror": {"number": 1, "location_description": "It's usually above the sink, so it should be on the top of the sink"},
        "green sink": {"number": 1, "location_description": "It's mentioned after the mirror, so it should be below the mirror"}
        }
Layout Dict: {"pink bathroom-0": (0, 0, 256, 256), "mirror-0": (90, 10, 166, 80), "green sink-0": (90, 90, 166, 180)}
layout:{'pink bathroom-0': [0, 0, 256, 256], 'mirror-0': [90, 10, 166, 80], 'green sink-0': [90, 90, 166, 180]}
result saved to ./layout_images/color6/A pink bathroom with a mirror and a green sink..png
'''


def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    # Add arguments
    parser.add_argument('--data_path', type=str, default= './T2I-CompBench_dataset', help='Path to prompts dataset main folder')
    parser.add_argument('--save_path', type=str, default='./layout', help='Path to main folder save layout.txt generating by gpt')

    # Parse the arguments
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    file_names = os.listdir(args.data_path)
    all_txt_file = [file for file in file_names if file.endswith('_val.txt')]

    for txt_idx, txt_file_path in enumerate(all_txt_file):
        save_sub_folder = txt_file_path.split(".")[0]+".txt"
        prompt = []
        # Getting prompt
        with open(os.path.join(args.data_path, txt_file_path), "r") as f:
            lines = f.readlines()
            for line in lines:
                prompt.append(line.replace("\n", "").replace(".", ""))

        for p_idx, caption in enumerate(prompt):
            iter = 0
            while True:
                try:
                    layout = gpt2layout(caption)
                    keys = list(set([key.split('-')[0] for key in layout.keys()]))
                    # Check if each key exists in the caption
                    # (if there is no key that is not in the caption but in the layout, you can exit)
                    if any(key in caption for key in keys):
                        break
                    else:
                        if iter <= 5:
                            iter += 1
                except Exception as e:
                    print(f"An error occurred: {e}")
                    time.sleep(1)

            with open(os.path.join(args.save_path, save_sub_folder), "w") as f:
                f.write(f"{caption}::::{layout}\n")



if __name__ == '__main__':
    main()
