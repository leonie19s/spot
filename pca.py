"""
adapted from: https://github.com/ShirAmir/dino-vit-features/tree/main

"""


import argparse
import random
import PIL.Image
from matplotlib import pyplot as plt
import numpy
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from typing import List, Tuple
from torchvision.transforms import ToPILImage
import os


save_dir = "pca" + os.sep
os.makedirs(save_dir, exist_ok=True)
def pca(features, img_batch, n_components: int = 4) -> List[Tuple[Image.Image, numpy.ndarray]]:
    """
     finding pca of a set of images.
    :param features: one feature layer, expected to be of shape [B, C, P]
    :img batch: batch of input images, expected to be of shape [B, H, W]
    :param n_components: number of pca components to produce.
   
    :return: a list of lists containing an image and its principal components.
    """
    to_pil = ToPILImage()
    img_pil_list = []
    for i in range(img_batch.shape[0]):
        img = img_batch[i]  # shape: [H, W]
       # img = img.unsqueeze(2)  # now [1, H, W]
        img = (img - img.min()) / (img.max() - img.min())*255
        img_np = img.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
        img_pil = Image.fromarray(img_np)
        img_pil_list.append(img_pil)
    pca_per_image = []
    for b in range(features.size(0)): # iterate over batch dimension
        feat = features[b]
        feat_np = feat.detach().cpu().numpy()
        pca = PCA(n_components=n_components).fit(feat_np)# resulting shape: [C, P]
        pca_descriptors = pca.transform(feat_np)
        pca_per_image.append(pca_descriptors)
    patch_dim = features.shape[1]
    results = [(pil_image, img_pca.reshape((int(np.sqrt(patch_dim))),(int(np.sqrt(patch_dim))) , n_components)) for
               (pil_image, img_pca) in zip(img_pil_list, pca_per_image)]
    return results

def shared_pca(pil_image, pca_images, save_dir):
    save_dir = Path(save_dir)
    #save_dir.mkdir(exist_ok=True, parents=True)
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    pil_image = pil_image.convert('L')
    axes = axes.flatten()
    axes[0].imshow(pil_image, cmap='gray')
    axes[0].axis('off')
    for i in range(len(pca_images)):
        pca_image = pca_images[i]
        comp = pca_image[:, :, -3:]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        axes[i + 1].imshow(comp_img)
        axes[i + 1].set_title(f'feature layer {i+1}')
        axes[i + 1].axis('off')
    for j in range(13, 15):
        axes[j].axis('off')
    plt.tight_layout()
    plt.savefig( save_dir / '')


def plot_pca(pil_image: Image.Image, pca_image: numpy.ndarray, save_dir: str, last_components_rgb: bool = True,
             save_resized=True, save_prefix: str = ''):
    """
    finding pca of a set of images.
    :param pil_image: The original PIL image.
    :param pca_image: A numpy tensor containing pca components of the image. HxWxn_components
    :param save_dir: if None than show results.
    :param last_components_rgb: If true save last 3 components as RGB image in addition to each component separately.
    :param save_resized: If true save PCA components resized to original resolution.
    :param save_prefix: optional. prefix to saving
    :return: a list of lists containing an image and its principal components.
    """
    pil_image = pil_image.convert('L')
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    pil_image_path = save_dir / f'{save_prefix}_orig_img.png'
    pil_image.save(pil_image_path)

    n_components = pca_image.shape[2]
    for comp_idx in range(n_components):
        comp = pca_image[:, :, comp_idx]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        comp_file_path = save_dir / f'{save_prefix}_{comp_idx}.png'
        pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
        if save_resized:
            pca_pil = pca_pil.resize(pil_image.size, resample=PIL.Image.NEAREST)
        pca_pil.save(comp_file_path)

    if last_components_rgb:
        comp_idxs = f"{n_components-3}_{n_components-2}_{n_components-1}"
        comp = pca_image[:, :, -3:]
        comp_min = comp.min(axis=(0, 1))
        comp_max = comp.max(axis=(0, 1))
        comp_img = (comp - comp_min) / (comp_max - comp_min)
        comp_file_path = save_dir / f'{save_prefix}_{comp_idxs}_rgb.png'
        pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
        if save_resized:
            pca_pil = pca_pil.resize(pil_image.size, resample=PIL.Image.NEAREST)
        pca_pil.save(comp_file_path)


def do_pca(layerwise_features, img_batch, counter):
    
    random_index_in_batch =  random.randint(0,img_batch.shape[0]-1)
    layer_pcas = []
    for idx, feat in enumerate(layerwise_features):
        layer_dir = os.path.join(save_dir, f"{counter}_layer_{idx}_b_{random_index_in_batch}" )
        result = pca(feat, img_batch)
        img, pca_res = result[random_index_in_batch]
        layer_pcas.append(pca_res)
        #plot_pca(img, pca_res,layer_dir)
    shared_pca(img, layer_pcas, layer_dir)
