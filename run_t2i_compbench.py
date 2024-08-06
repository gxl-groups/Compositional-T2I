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


def get_indices_to_alter(stable, prompt: str, need_input: bool, token: str = None) -> List[int]:
    token_idx_to_word = {idx: stable.tokenizer.decode(t)
                         for idx, t in enumerate(stable.tokenizer(prompt)['input_ids'])
                         if 0 < idx < len(stable.tokenizer(prompt)['input_ids']) - 1}
    if need_input and token is not None:
        pprint.pprint(token_idx_to_word)
        token_indices = int(input(f"Please enter the index position where the {token} is located(e.g. 2): "))
        return token_indices
    else:
        return token_idx_to_word


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


def get_token_indice(token, token_idx_to_word, token_indice, caption, stable):
    token = token.split("'")[0]
    token_plural = p.plural(token)  # 单词的复数形式
    find_indx = False
    for key, value in token_idx_to_word.items():
        if (value == token or value == token_plural) and int(key) not in token_indice:
            token_indice.append(int(key))
            find_indx = True
            break
    if not find_indx:
        token_indice.append(get_indices_to_alter(stable, caption, need_input=True, token=token))

    return find_indx, token_indice


def get_token_pof(token, doc):
    get_part_of_speech = False
    pof_index = None
    for doc_token in doc:
        if token == doc_token.text:
            if doc_token.pos_ in ["NOUN", "PROPN"]:
                pof_index = 1
                get_part_of_speech = True
            elif doc_token.pos_ in ["ADJ", "amod", "nmod", "compound",
                                    "npadvmod", "advmod", "acomp", 'relcl']:
                pof_index = 0
                get_part_of_speech = True
        if get_part_of_speech:
            break

    return pof_index, get_part_of_speech

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
    file_names = os.listdir(config.data_path)
    all_txt_file = [file for file in file_names if file.endswith('_val.txt')]

    for txt_file_path in all_txt_file:
        save_sub_folder = txt_file_path.split(".")[0]
        prompt = []
        # Getting prompt
        with open(config.data_path + txt_file_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                prompt.append(line.replace("\n", "").replace(".", ""))

        # Getting layout
        all_caption_layout = {}
        with open(config.layout_path + txt_file_path.split("/")[-1], "r") as f:
            lines = f.readlines()
            for line in lines:
                caption_key = line.split("::::")[0]
                layout_dict = line.split("::::")[-1].replace("\n", "")
                pattern = r"'(.*?)': \[(.*?)\]"
                matches = re.findall(pattern, layout_dict)
                layout_dict = {key: list(map(int, value.split(', '))) for key, value in matches}
                all_caption_layout[caption_key] = layout_dict

        for p_id, caption in enumerate(prompt):
            if (p_id >= 0):
                # get the caption's layout from txt_file(save the layout generating by gpt)
                layout = all_caption_layout[caption]
                layout_key = []
                for key, value in layout.items():
                    layout_key.append(key.split("-")[0])
                layout_key = list(set(layout_key))

                token_idx_to_word = get_indices_to_alter(stable, caption, need_input=False)
                doc = nlp(caption)
                bbox, token_indice, part_of_speech = [], [], []
                for eve_tokens in layout_key:
                    tokens = eve_tokens.split(" ")
                    for token in tokens:
                        # Getting token_indices: token_pof =1 if noun else 0
                        token_pof, get_pof = get_token_pof(token, doc)
                        # If this token is what we want to pay attention to
                        if get_pof:
                            part_of_speech.append(token_pof)
                            # Find the word's index in tokenizer sequence
                            find_indx, token_find_index = get_token_indice(token, token_idx_to_word, token_indice,
                                                                           caption, stable)
                            if find_indx:
                                token_indice = token_find_index
                                word_box = [layout[key] for key, value in layout.items() if eve_tokens in key]
                                bbox.append(word_box[0] if len(word_box) == 1 else word_box)

                # If All the parameters required by the prompt are available, then generate
                if len(token_indice) == len(bbox) == len(part_of_speech) and len(token_indice) != 0:
                    sorted_tokens = sorted(zip(token_indice, part_of_speech, bbox))
                    token_indice, part_of_speech, bbox = zip(*sorted_tokens)
                    bbox = bbox_resize(bbox, token_indice, new_size=config.attention_res)

                    seed = config.seeds
                    print(f"Seed: {seed}")
                    print(f"PROMPT:{caption}")
                    print(f"image save path is {config.output_path}")

                    g = torch.Generator('cuda').manual_seed(seed)
                    controller = AttentionStore()
                    for index in tqdm(range(0, config.per_num)):
                        image = run_on_prompt(prompt=caption,
                                              model=stable,
                                              controller=controller,
                                              token_indices=token_indice,
                                              bbox=bbox,
                                              part_of_speech=part_of_speech,
                                              seed=g,
                                              config=config)
                        prompt_output_path = config.output_path / save_sub_folder
                        prompt_output_path.mkdir(exist_ok=True, parents=True)
                        image.save(prompt_output_path / f'{caption}_{p_id}_{index}.png')


if __name__ == '__main__':
    main()
