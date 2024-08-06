import math
from typing import List
from PIL import Image
import cv2
import numpy as np
import torch
import os
import imageio
from utils import ptp_utils
from utils.ptp_utils import AttentionStore, aggregate_attention


def show_cross_attention(prompt: str,
                         attention_store: AttentionStore,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         from_where: List[str],
                         select: int = 0,
                         radius:int=20,
                         orig_image=None):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select).detach().cpu()
    images = []

    # show spatial attention for indices of tokens to strengthen
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            image = show_image_relevance(image, orig_image,radius=radius)
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
            images.append(image)

    ptp_utils.view_images(np.stack(images, axis=0))


def save_cross_attention(prompt: str,
                         attention_store: AttentionStore,
                         tokenizer,
                         indices_to_alter: List[int],
                         res: int,
                         from_where: List[str],
                         select: int = 0,
                         orig_image=None,
                         output_path="./visualization_outputs/crossAttn",
                         iter_num=51):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select).detach().cpu()
    images = []

    # show spatial attention for indices of tokens to strengthen
    max_position_per_index = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        if i in indices_to_alter:
            ii = torch.argmax(image)
            max_position_per_index.append(ii)
            image, image_16 = show_image_relevance(image,
                                                   orig_image, is_self=True, x=ii // 16,
                                                   y=ii % 16,radius=20)  # Retro: the key step, process attention map and original image
            image = image.astype(np.uint8)
            image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])) + f" {ii // 16 + 1},{ii % 16 + 1}")
            images.append(image)


    pil_img = ptp_utils.view_images(np.stack(images, axis=0), display_image=False)
    os.makedirs(output_path + "/heatmap_" + prompt.replace(" ", "_"), exist_ok=True)
    pil_img.save(output_path + "/heatmap_" + prompt.replace(" ", "_") + f"/iter_{iter_num:02d}.jpg")
    if iter_num == 50:
        image_folder = output_path + "/heatmap_" + prompt.replace(" ", "_")
        output_gif_file = output_path + "/heatmap_" + prompt.replace(" ", "_") + "/output.gif"
        output_mov_file = output_path + "/heatmap_" + prompt.replace(" ", "_") + "/output.mov"
        images = [imageio.imread(os.path.join(image_folder, image_file)) for image_file in
                  sorted(os.listdir(image_folder)) if image_file.endswith('.jpg')]
        imageio.mimsave(output_gif_file, images, duration=0.5)  # 设置 duration 参数以控制每帧之间的时间间隔
        imageio.mimsave(output_mov_file, images, 'ffmpeg', fps=3)  # 设置 fps 参数以控制每秒帧数
    return max_position_per_index


def save_self_attention(prompt: str,
                        attention_store: AttentionStore,
                        tokenizer,
                        indices_to_alter: List[int],
                        res: int,
                        from_where: List[str],
                        select: int = 0,
                        orig_image=None,
                        output_path="./visualization_outputs/selfAttn",
                        iter_num=51):
    tokens = tokenizer.encode(prompt)
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).detach().cpu()
    images = []
    # images_16 = []
    # show spatial attention for indices of tokens to strengthen
    for i in indices_to_alter:
        image = attention_maps[:, :, i]
        image, image_16 = show_image_relevance(image,
                                               orig_image, is_self=True, x=i // 16,
                                               y=i % 16)  # Retro: the key step, process attention map and original image
        image = image.astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((res ** 2, res ** 2)))
        image = ptp_utils.text_under_image(image, f"{i} {i // 16 + 1},{i % 16 + 1}")
        images.append(image)


    pil_img = ptp_utils.view_images(np.stack(images, axis=0), display_image=False)
    # pil_img_16 = ptp_utils.view_images(np.stack(images_16, axis=0), display_image=False)
    os.makedirs(output_path + "/heatmap_" + prompt.replace(" ", "_"), exist_ok=True)
    pil_img.save(output_path + "/heatmap_" + prompt.replace(" ", "_") + f"/iter_{iter_num:02d}.jpg")
    # pil_img_16.save(output_path + "/heatmap_" + prompt.replace(" ", "_") + f"/16_iter_{iter_num:02d}.jpg")

    if iter_num == 50:
        image_folder = output_path + "/heatmap_" + prompt.replace(" ", "_")
        output_gif_file = output_path + "/heatmap_" + prompt.replace(" ", "_") + "/output.gif"
        output_mov_file = output_path + "/heatmap_" + prompt.replace(" ", "_") + "/output.mov"
        images = [imageio.imread(os.path.join(image_folder, image_file)) for image_file in
                  sorted(os.listdir(image_folder)) if image_file.endswith('.jpg')]
        # Set the duration parameter to control the time interval between each frame.
        imageio.mimsave(output_gif_file, images, duration=0.5)
        # Set the fps parameter to control the number of frames per second
        imageio.mimsave(output_mov_file, images, 'ffmpeg', fps=3)


def show_image_relevance(image_relevance, image: Image.Image,
                         relevnace_res=16, is_self=False, x=0,
                         y=0,radius=20):  # Retro: image_relevance{Tensor:(16,16)} rensor([[0.0006,....],..])
    # create heatmap from mask on image
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam

    # Retro add:
    image_relevance_to_show = image_relevance.numpy()
    image_relevance_to_show = (image_relevance_to_show - image_relevance_to_show.min()) / (
            image_relevance_to_show.max() - image_relevance_to_show.min())
    image_relevance_to_show = np.uint8(255 * image_relevance_to_show)
    # image_relevance_to_show = Image.fromarray(image_relevance_to_show)
    # image_relevance_to_show.save("./visualization_outputs/heatmap16.jpg")
    image_relevance_to_show = image_relevance_to_show.reshape(image_relevance_to_show.shape[0], -1, 1)

    image = image.resize((relevnace_res ** 2, relevnace_res ** 2))  # {Image} 256*256
    image = np.array(image)  # Retro {ndarray:(256,256,3)} [[[16,21,25],[18,22,26],...],...]

    image_relevance = image_relevance.reshape(1, 1, image_relevance.shape[-1],
                                              image_relevance.shape[-1])  # Retro: {Tensor:(1,1,16,16)}
    image_relevance = image_relevance.cuda(2)  # because float16 precision interpolation is not supported on cpu
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=relevnace_res ** 2,
                                                      mode='bilinear')  # Retro: {Tensor:(1,1,256,256)}
    image_relevance = image_relevance.cpu()  # send it back to cpu
    image_relevance = (image_relevance - image_relevance.min()) / (
            image_relevance.max() - image_relevance.min())  # Retro: normalized->[0, 1]
    image_relevance = image_relevance.reshape(relevnace_res ** 2, relevnace_res ** 2)  # Retro: {Tensor:(256,256)}
    image = (image - image.min()) / (image.max() - image.min())  # Retro: normalized->[0,1] {ndarray:(256,256,3)}

    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    if is_self:
        mapped_x = int((x / 15) * 255)
        mapped_y = int((y / 15) * 255)
        radius = radius
        color = (0, 0, 255)
        cv2.circle(vis, (mapped_y, mapped_x), radius, color, -1)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)

    return vis, image_relevance_to_show


def get_image_grid(images: List[Image.Image]) -> Image:
    num_images = len(images)
    cols = int(math.ceil(math.sqrt(num_images)))
    rows = int(math.ceil(num_images / cols))
    width, height = images[0].size
    grid_image = Image.new('RGB', (cols * width, rows * height))
    for i, img in enumerate(images):
        x = i % cols
        y = i // cols
        grid_image.paste(img, (x * width, y * height))
    return grid_image
