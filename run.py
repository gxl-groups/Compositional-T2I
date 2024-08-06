import pprint
from typing import List

import pyrallis
import torch
from PIL import Image, ImageDraw
import random
import inflect
import re
import os
import time
import math
import spacy

from config import RunConfig
from pipeline import AttendAndExcitePipeline
from utils import ptp_utils, vis_utils
from utils.ptp_utils import AttentionStore
import warnings
from tqdm import tqdm
from utils.gpt2layout import gpt2layout

warnings.filterwarnings("ignore", category=UserWarning)

nlp = spacy.load("en_core_web_trf")
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
p = inflect.engine()


def load_model(config: RunConfig):
    sd_model_version = "stabilityai/stable-diffusion-2-1-base" if config.sd_2_1 else "CompVis/stable-diffusion-v1-4"
    if config.model_path is not None:
        model_path = config.model_path
    else:
        model_path = sd_model_version

    stable = AttendAndExcitePipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)

    return stable


def get_indices_to_alter(stable, prompt: str) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    pprint.pprint(token_idx_to_word)
    token_indices = input("Please enter the a comma-separated list indices of the tokens you wish to "
                          "alter (e.g., 2,5): ")
    token_indices = [int(i) for i in token_indices.split(",")]
    print(f"Altering tokens: {[token_idx_to_word[i] for i in token_indices]}")
    return token_indices

def get_part_of_speech(prompt):
    doc = nlp(prompt)
    print_dict = {}
    for sub_doc in doc:
        print_dict[sub_doc.text] = sub_doc.pos_
    print(print_dict)
    print("if the word's pos is NOUN or PROPN, the input value will be 0 \n"
          "else the pos in [ADJ, amod, nmod, compound, npadvmod, advmod, acomp, relcl ] the value will be 1")
    part_of_speech = input("Please enter the list part-of-speech of the tokens you wish to pay attention to "
                          "alter (e.g. 0,1,0,1): ")
    part_of_speech = [int(i) for i in part_of_speech.split(",")]
    return part_of_speech

def get_box(prompt):
    iter = 0
    while True:
        try:
            layout = gpt2layout(prompt)
            keys = [key.split('-')[0] for key in layout.keys()]
            # Check if each key exists in the caption
            # (if there is no key that is not in the caption but in the layout, you can exit)
            if not any(key not in prompt for key in keys):
                break
            else:
                if iter <= 5:
                    iter += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(1)
    print(f"Pless input the box list you want. A box that can only input one word at a time (e.g. 10,10,30,30)))\n"
          f"If a word has multi box, please enter the coordinates of multiple boxes continuously (e.g. 10,10,30,30,40,40,50,50)\n"
          f"If you want to finish input box, pless input 'finish'\n"
          f"layout dict: {layout}\n ")
    bbox=[]
    while True:
        line = input("Enter a box : ")
        if "finish" in line:
            break
        box = [int(i) for i in line.split(",")]
        if len(box) > 4:
            box = [box[i:i + 4] for i in range(0, len(box), 4)]

        bbox.append(box)
    return bbox

def get_prompt():
    prompt = input("Please enter the prompt you want(e.g. a red dog and a blue bowel):")
    return prompt

def run_on_prompt(prompt: List[str],
                  model: AttendAndExcitePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  bbox: List[List[int]],
                  part_of_speech: List[List[int]],
                  seed: torch.Generator,
                  config: RunConfig) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)
    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    bbox=bbox,
                    part_of_speech=part_of_speech,
                    soft_mask_rate=config.soft_mask_rate,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    num_inference_steps=config.n_inference_steps,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    att_vis=config.attention_vis,
                    attention_vis_file=config.attention_vis_file,
                    att_vis_timestep=config.att_vis_timestep)
    image = outputs.images[0]
    return image

def box_reize(box, old_size=256, new_size=16):
    if all(element == 0 for element in box):
        x_random = random.choice(range(new_size - 1))
        box = [x_random, x_random, x_random + 1, x_random + 1]
    else:
        box = [int(min(max(value, 0), old_size - 1) * (new_size / old_size)) for value in
                                  box]
    return box

def bbox_resize(bbox, token_indices, old_size=256, new_size=16):
    new_bbox = [[0, 0, old_size, old_size] for _ in range(77)]
    for i in range(len(token_indices)):
        new_bbox[token_indices[i]] = bbox[i]

    for idx, box in enumerate(new_bbox):
        if isinstance(box[0], list):
            for sub_idx, sub_box in enumerate(box):
                new_bbox[idx][sub_idx] = box_reize(sub_box)
        else:
            new_bbox[idx] = box_reize(box)

    return new_bbox

@pyrallis.wrap()
def main(config: RunConfig):
    stable = load_model(config)
    prompt = get_prompt() if config.prompt is None else config.prompt
    token_indice = get_indices_to_alter(stable, prompt) if config.token_indices is None else config.token_indices
    part_of_speech = get_part_of_speech(prompt) if config.part_of_speech is None else config.part_of_speech
    bbox = get_box(prompt) if config.bbox is None else (config.bbox if len(config.bbox)<=4 else [config.bbox[i:i + 4] for i in range(0, len(config.bbox), 4)])

    bbox = bbox_resize(bbox,token_indice,config.attention_res)

    seed = config.seeds
    print(f"Seed: {seed}")
    print(f"PROMPT:{prompt}")
    print(f"image save path is {config.output_path}")

    g = torch.Generator('cuda').manual_seed(seed)
    controller = AttentionStore()
    for index in tqdm(range(0, config.per_num)):
        image = run_on_prompt(prompt=prompt,
                              model=stable,
                              controller=controller,
                              token_indices=token_indice,
                              bbox=bbox,
                              part_of_speech=part_of_speech,
                              seed=g,
                              config=config)
        prompt_output_path = config.output_path
        prompt_output_path.mkdir(exist_ok=True, parents=True)
        image.save(prompt_output_path / f'{prompt}_{index}.png')


if __name__ == '__main__':
    main()
