''' Based on SLATE and OCLF libraries:
https://github.com/singhgautam/slate/blob/master/utils.py
https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/utils/masking.py
'''
import math
import os
import random
import warnings
import argparse
from matplotlib import cm, pyplot as plt
import numpy as np
from typing import Optional
from PIL import ImageFilter
from collections import OrderedDict
from einops import rearrange, repeat
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import generate_binary_structure
import matplotlib.colors as mcolors
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
from swin import build_swin_model
from scipy import interpolate

binary_structure = generate_binary_structure(2,2)

colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080','#C56932',
'#b7a58c', '#3a627d', '#9abc15', '#54810c', '#a7389c', '#687253', '#61f584', '#9a17d4', '#52b0c1', '#21f5b4', '#a2856c', '#9b1c34', '#4b1062', '#7cf406', '#0b1f63']

def gumbel_max(logits, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = logits + gumbels

    return gumbels.argmax(dim)


def gumbel_softmax(logits, tau=1., hard=False, dim=-1):

    eps = torch.finfo(logits.dtype).tiny

    gumbels = -(torch.empty_like(logits).exponential_() + eps).log()
    gumbels = (logits + gumbels) / tau

    y_soft = F.softmax(gumbels, dim)

    if hard:
        index = y_soft.argmax(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.)
        return y_hard - y_soft.detach() + y_soft
    else:
        return y_soft


def log_prob_gaussian(value, mean, std):

    var = std ** 2
    if isinstance(var, float):
        return -0.5 * (((value - mean) ** 2) / var + math.log(var) + math.log(2 * math.pi))
    else:
        return -0.5 * (((value - mean) ** 2) / var + var.log() + math.log(2 * math.pi))


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):

    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=False, weight_init='kaiming')
        self.weight = nn.Parameter(torch.ones(out_channels))
        self.bias = nn.Parameter(torch.zeros(out_channels))


    def forward(self, x):

        x = self.m(x)
        return F.relu(F.group_norm(x, 1, self.weight, self.bias))


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):

    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gru_cell(input_size, hidden_size, bias=True):

    m = nn.GRUCell(input_size, hidden_size, bias)

    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)

    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

    return m

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

inv_normalize = transforms.Compose([transforms.Normalize((0., 0., 0.), (1/0.229, 1/0.224, 1/0.225)),
                                    transforms.Normalize((-0.485, -0.456, -0.406), (1, 1, 1))])


def pairwise_IoU(pred_mask, gt_mask):
    pred_mask = repeat(pred_mask, "... n c -> ... 1 n c")
    gt_mask = repeat(gt_mask, "... n c -> ... n 1 c")
    intersection = pred_mask & gt_mask
    union = pred_mask | gt_mask
    union_sum = torch.clamp(union.sum(-1).float(), min=0.000001) # to avoid division by zero.
    iou = intersection.sum(-1) / union_sum
    return iou


def pairwise_IoU_efficient(pred_mask, gt_mask):
    intersection = torch.einsum("bkj,bij->bki", gt_mask, pred_mask)
    union = gt_mask.sum(dim=2).unsqueeze(dim=2) + pred_mask.sum(dim=2).unsqueeze(dim=1) - intersection
    iou = intersection / torch.clamp(union, min=0.000001) # to avoid division by zero.
    return iou


def compute_IoU(pred_mask, gt_mask):
    # assumes shape: batch_size, set_size, channels
    is_padding = (gt_mask == 0).all(-1)

    # discretized 2d mask for hungarian matching
    pred_mask_id = torch.argmax(pred_mask, -2)
    num_slots = pred_mask.size(1)
    if num_slots < gt_mask.size(1): # essentially padding the pred_mask if num_slots < gt_mask.size(1)
        num_slots = gt_mask.size(1)
    pred_mask_disc = rearrange(
        F.one_hot(pred_mask_id, num_slots).to(torch.float32), "b c n -> b n c"
    )

    # treat as if no padding in gt_mask
    pIoU = pairwise_IoU_efficient(pred_mask_disc.float(), gt_mask.float())
    #pIoU = pairwise_IoU(pred_mask_disc.bool(), gt_mask.bool())
    pIoU_inv = 1 - pIoU
    pIoU_inv[is_padding] = 1e3
    pIoU_inv_ = pIoU_inv.detach().cpu().numpy()

    # hungarian matching
    indices = np.array([linear_sum_assignment(p) for p in pIoU_inv_])
    indices_ = pred_mask.size(1) * indices[:, 0] + indices[:, 1]
    indices_ = torch.from_numpy(indices_).to(device=pred_mask.device)
    IoU = torch.gather(rearrange(pIoU, "b n m -> b (n m)"), 1, indices_)
    mIoU = (IoU * ~is_padding).sum(-1) / torch.clamp((~is_padding).sum(-1), min=0.000001) # to avoid division by zero.
    return mIoU

def att_matching(attention_1, attention_2):

    batch_size, slots, height, width = attention_1.shape
    
    mask_1 = torch.nn.functional.one_hot(attention_1.argmax(1).reshape(batch_size,-1), num_classes=slots).to(torch.float32).permute(0,2,1)
    mask_2 = torch.nn.functional.one_hot(attention_2.argmax(1).reshape(batch_size,-1), num_classes=slots).to(torch.float32).permute(0,2,1)
    
    # assumes shape: batch_size, set_size, channels
    is_padding = (mask_2 == 0).all(-1)

    # discretized 2d mask for hungarian matching
    pred_mask_1_id = torch.argmax(mask_1, -2)
    pred_mask_1_disc = rearrange(
        F.one_hot(pred_mask_1_id, mask_1.size(1)).to(torch.float32), "b c n -> b n c"
    )

    # treat as if no padding in mask_2
    pIoU = pairwise_IoU_efficient(pred_mask_1_disc.float(), mask_2.float())
    pIoU_inv = 1 - pIoU
    pIoU_inv[is_padding] = 1e3
    pIoU_inv_ = pIoU_inv.detach().cpu().numpy()

    # hungarian matching
    indices = np.array([linear_sum_assignment(p)[1] for p in pIoU_inv_])
    #attention_2_permuted = torch.stack([x[indices[n]] for n, x in enumerate(attention_2)],dim=0)

    pIoU = pIoU.detach().cpu().numpy()
    matching_scores = np.array([[pIoU[b][i,j] for i,j in enumerate(indices[b])] for b in range(batch_size)])
    return indices, matching_scores

def trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def exp_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, decay_rate=5, plateau_epochs=4):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    
    # Warmup phase
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    
    plateau_iters = plateau_epochs * niter_per_ep  # Number of iterations in the plateau phase
    plateau_schedule = np.full(plateau_iters, base_value)  # Keep constant at base_value

    decay_iters = epochs * niter_per_ep - warmup_iters - plateau_iters
    iters = np.arange(decay_iters)
    
    # Exponential decay phase
    decay_schedule = final_value + (base_value - final_value) * np.exp(-decay_rate * iters / decay_iters)

    # Combine all schedules
    schedule = np.concatenate((warmup_schedule, plateau_schedule, decay_schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
        
#Copied from https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/utils/masking.py
"""Utilities related to masking."""

class CreateSlotMask(nn.Module):
    """Module intended to create a mask that marks empty slots.
    Module takes a tensor holding the number of slots per batch entry, and returns a binary mask of
    shape (batch_size, max_slots) where entries exceeding the number of slots are masked out.
    """

    def __init__(self, max_slots: int):
        super().__init__()
        self.max_slots = max_slots

    def forward(self, n_slots: torch.Tensor) -> torch.Tensor:
        (batch_size,) = n_slots.shape

        # Create mask of shape B x K where the first n_slots entries per-row are false, the rest true
        indices = torch.arange(self.max_slots, device=n_slots.device)
        masks = indices.unsqueeze(0).expand(batch_size, -1) >= n_slots.unsqueeze(1)

        return masks

#Copied from https://github.com/amazon-science/object-centric-learning-framework/blob/main/ocl/utils/masking.py
class CreateRandomMaskPatterns(nn.Module):
    """Create random masks.
    Useful for showcasing behavior of metrics.
    """

    def __init__(self, pattern: str, n_slots: Optional[int] = None, n_cols: int = 2):
        super().__init__()
        if pattern not in ("random", "blocks"):
            raise ValueError(f"Unknown pattern {pattern}")
        self.pattern = pattern
        self.n_slots = n_slots
        self.n_cols = n_cols

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        if self.pattern == "random":
            rand_mask = torch.rand_like(masks)
            return rand_mask / rand_mask.sum(1, keepdim=True)
        elif self.pattern == "blocks":
            n_slots = masks.shape[1] if self.n_slots is None else self.n_slots
            height, width = masks.shape[-2:]
            new_masks = torch.zeros(
                len(masks), n_slots, height, width, device=masks.device, dtype=masks.dtype
            )
            blocks_per_col = int(n_slots // self.n_cols)
            remainder = n_slots - (blocks_per_col * self.n_cols)
            slot = 0
            for col in range(self.n_cols):
                rows = blocks_per_col if col < self.n_cols - 1 else blocks_per_col + remainder
                for row in range(rows):
                    block_width = math.ceil(width / self.n_cols)
                    block_height = math.ceil(height / rows)
                    x = col * block_width
                    y = row * block_height
                    new_masks[:, slot, y : y + block_height, x : x + block_width] = 1
                    slot += 1
            assert torch.allclose(new_masks.sum(1), torch.ones_like(masks[:, 0]))
            return new_masks
        
def spiral_pattern(A, how = 'left_top'):
    
    out = []
    
    if how == 'left_top':
        A = np.array(A)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'top_left':
        A = np.rot90(np.fliplr(np.array(A)), k=1)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'right_top':
        A = np.fliplr(np.array(A))
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'top_right':
        A = np.rot90(np.array(A), k=1)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'right_bottom':
        A = np.rot90(np.array(A), k=2)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'bottom_right':
        A = np.fliplr(np.rot90(np.array(A), k=1))
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'left_bottom':
        A = np.rot90(np.fliplr(np.array(A)), k=2)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
            
    if how == 'bottom_left':
        A = np.rot90(np.array(A), k=3)
        while(A.size):
            out.append(A[0])        # take first row
            A = A[1:].T[::-1]       # cut off first row and rotate counterclockwise
    return np.concatenate(out)


def visualize(image, true_mask, pred_dec_mask, rgb_dec_attns, pred_default_mask, rgb_default_attns, N=8):
    _, _, H, W = image.shape
    
    rgb_pred_dec_mask = (torch.stack([draw_segmentation_masks((image[idx]*255).to(torch.uint8).cpu(), 
                                        masks= torch.nn.functional.one_hot(pred_dec_mask[idx]).permute(2,0,1).to(torch.bool).cpu(), 
                                        alpha=.5,
                                        colors = colors) for idx in range(image.shape[0])])/255.)
    
    rgb_pred_default_mask = (torch.stack([draw_segmentation_masks((image[idx]*255).to(torch.uint8).cpu(), 
                                    masks= torch.nn.functional.one_hot(pred_default_mask[idx]).permute(2,0,1).to(torch.bool).cpu(), 
                                    alpha=.5,
                                    colors = colors) for idx in range(image.shape[0])])/255.)

    _, true_mask_unique = torch.unique(true_mask,return_inverse=True)
    rgb_true_mask = (torch.stack([draw_segmentation_masks((image[idx]*255).to(torch.uint8).cpu(), 
                                masks= torch.nn.functional.one_hot(true_mask_unique[idx]).permute(2,0,1).to(torch.bool).cpu(), 
                                alpha=.5,
                                colors = colors) for idx in range(image.shape[0])])/255.)
    
    image = image[:N].expand(-1, 3, H, W).unsqueeze(dim=1).cpu()
    rgb_default_attns = rgb_default_attns[:N].expand(-1, -1, 3, H, W).cpu()
    rgb_dec_attns = rgb_dec_attns[:N].expand(-1, -1, 3, H, W).cpu()

    rgb_true_mask = rgb_true_mask.unsqueeze(dim=1).cpu()
    rgb_pred_default_mask = rgb_pred_default_mask.unsqueeze(dim=1).cpu()
    rgb_pred_dec_mask = rgb_pred_dec_mask.unsqueeze(dim=1).cpu()

    return torch.cat((image, rgb_true_mask, rgb_pred_dec_mask, rgb_dec_attns, rgb_pred_default_mask, rgb_default_attns), dim=1).view(-1, 3, H, W)


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def remap_pretrained_keys_swin(model, checkpoint_model):
    state_dict = model.state_dict()
    
    # Geometric interpolation when pre-trained patch size mismatch with fine-tuned patch size
    all_keys = list(checkpoint_model.keys())
    for key in all_keys:
        if "relative_position_bias_table" in key:
            relative_position_bias_table_pretrained = checkpoint_model[key]
            relative_position_bias_table_current = state_dict[key]
            L1, nH1 = relative_position_bias_table_pretrained.size()
            L2, nH2 = relative_position_bias_table_current.size()
            if nH1 != nH2:
                print(f"Error in loading {key}, passing......")
            else:
                if L1 != L2:
                    print(f"{key}: Interpolate relative_position_bias_table using geo.")
                    src_size = int(L1 ** 0.5)
                    dst_size = int(L2 ** 0.5)

                    def geometric_progression(a, r, n):
                        return a * (1.0 - r ** n) / (1.0 - r)

                    left, right = 1.01, 1.5
                    while right - left > 1e-6:
                        q = (left + right) / 2.0
                        gp = geometric_progression(1, q, src_size // 2)
                        if gp > dst_size // 2:
                            right = q
                        else:
                            left = q

                    # if q > 1.090307:
                    #     q = 1.090307

                    dis = []
                    cur = 1
                    for i in range(src_size // 2):
                        dis.append(cur)
                        cur += q ** (i + 1)

                    r_ids = [-_ for _ in reversed(dis)]

                    x = r_ids + [0] + dis
                    y = r_ids + [0] + dis

                    t = dst_size // 2.0
                    dx = np.arange(-t, t + 0.1, 1.0)
                    dy = np.arange(-t, t + 0.1, 1.0)

                    print("Original positions = %s" % str(x))
                    print("Target positions = %s" % str(dx))

                    all_rel_pos_bias = []

                    for i in range(nH1):
                        z = relative_position_bias_table_pretrained[:, i].view(src_size, src_size).float().numpy()
                        f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                        all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                            relative_position_bias_table_pretrained.device))

                    new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                    checkpoint_model[key] = new_rel_pos_bias

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [k for k in checkpoint_model.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del checkpoint_model[k]

    # delete relative_coords_table since we always re-init it
    relative_coords_table_keys = [k for k in checkpoint_model.keys() if "relative_coords_table" in k]
    for k in relative_coords_table_keys:
        del checkpoint_model[k]

    # re-map keys due to name change
    rpe_mlp_keys = [k for k in checkpoint_model.keys() if "rpe_mlp" in k]
    for k in rpe_mlp_keys:
        checkpoint_model[k.replace('rpe_mlp', 'cpb_mlp')] = checkpoint_model.pop(k)

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in checkpoint_model.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del checkpoint_model[k]

    return checkpoint_model

def load_swin_encoder(model):
    checkpoint = torch.load("/visinf/home/vilab01/spot/local/swinv2_base_22k_500k.pth", map_location='cpu')
    checkpoint_model = checkpoint['model']
    checkpoint = remap_pretrained_keys_swin(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    del checkpoint
    torch.cuda.empty_cache()

   
    
def load_pretrained_encoder(model, pretrained_weights, prefix=None):
    if pretrained_weights:
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrained_weights)
        if "state_dict" in checkpoint:
            checkpoint_model = checkpoint["state_dict"]
        elif 'target_encoder' in checkpoint:
            checkpoint_model = checkpoint["target_encoder"]
        elif 'model' in checkpoint:
            checkpoint_model = checkpoint["model"]

        checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint_model.items()}

        if prefix is not None:
            checkpoint = checkpoint_model
            checkpoint_model = OrderedDict()
            # Keep only the parameters/buffers of the ViT encoder.
            all_keys = list(checkpoint.keys())
            counter = 0
            for key in all_keys:
                if key.startswith(prefix):
                    counter += 1
                    new_key = key[len(prefix):]
                    print(f"\t #{counter}: {key} ==> {new_key}")
                    checkpoint_model[new_key] = checkpoint[key]

        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        print("Model = %s" % str(model))
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        assert len(set(msg.missing_keys)) == 0


def reduce_dataset(dataset, percentage):

    # Generate a list of indices that are to be kept and randomly shuffle
    reduced_size = int(len(dataset) * percentage)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    
    # Select the first reduced_size indices
    subset_indices = indices[:reduced_size]

    # Create torch.utils.data.subset and return
    return Subset(dataset, subset_indices)


def check_for_nan_inf(module, input, output):
    """
    input: input tensor(s) to the module
    output: output tensor(s) of the module
    -> we only check the output, because the input is the output of the previous module

    """
    def inspect_tensor(tensor, context):
        if isinstance(tensor, torch.Tensor):
            nan_mask = torch.isnan(tensor)
            inf_mask = torch.isinf(tensor)
            if nan_mask.any() or inf_mask.any():
                print(f"Warning: Detected NaN or Inf in {context} of module {module.__class__.__name__}")
                print(f"NaN count: {nan_mask.sum().item()}, Inf count: {inf_mask.sum().item()}")

    # Check outputs (handle single tensor, list, or tuple)
    if isinstance(output, torch.Tensor):
        inspect_tensor(output, "output")
    elif isinstance(output, (list, tuple)):
        for idx, out in enumerate(output):
            inspect_tensor(out, f"output {idx}")

def save_layer_attn_images(base_path, attn_masks_list, image, fused_mask, batch_index = 1, upsample_size=960, iteration=0, n_slots = 6, gt=None):

    def create_custom_cmap(num_colors):
        if not (6 <= num_colors <= 24):
            raise ValueError("Number of colors must be between 6 and 24.")
        colors_6 = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b"   # Brown
        ]
        if num_colors == 6:
            return mcolors.ListedColormap(colors_6, name=f"custom_{num_colors}")
        colors_7 = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2"   # Pink
        ]
        if num_colors == 7:
            return mcolors.ListedColormap(colors_7, name=f"custom_{num_colors}")

        base_colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Lime
            "#17becf",  # Cyan
            "#aec7e8",  # Light Blue
            "#ffbb78",  # Light Orange
            "#98df8a",  # Light Green
            "#ff9896",  # Light Red
            "#c5b0d5",  # Light Purple
            "#c49c94",  # Light Brown
            "#f7b6d2",  # Light Pink
            "#c7c7c7",  # Light Gray
            "#dbdb8d",  # Light Lime
            "#9edae5",  # Light Cyan
            "#393b79",  # Dark Blue
            "#637939",  # Dark Green
            "#8c6d31",  # Dark Orange
            "#843c39"   # Dark Red
        ]

        colors = base_colors[:num_colors]
        return mcolors.ListedColormap(colors, name=f"custom_{num_colors}")
    
    # preprocess
    attn_mask_list = [attn_mask[batch_index] for attn_mask in attn_masks_list]
    h_w = int(math.sqrt(attn_mask_list[0].shape[0]))
    attn_mask_list_bi = [attn_mask.reshape(h_w, h_w, n_slots) for attn_mask in attn_mask_list]
    attn_mask_upsampled = [F.interpolate(
        attn_mask.permute(2, 0, 1).unsqueeze(0), 
        size=(upsample_size, upsample_size),  
        mode='bilinear'
    ).squeeze(0).permute(1, 2, 0)  for attn_mask in attn_mask_list_bi]
    attn_masks_np = [attn_mask.clone().detach().cpu().numpy() for attn_mask in attn_mask_upsampled]
    fused_mask = fused_mask[batch_index].reshape(h_w, h_w, n_slots)
    fused_mask = F.interpolate(
        fused_mask.permute(2, 0, 1).unsqueeze(0), 
        size=(upsample_size, upsample_size),  
        mode='bilinear'
    ).squeeze(0).permute(1, 2, 0)
    fused_mask = fused_mask.clone().detach().cpu().numpy()
    image = image[batch_index].clone().detach().cpu()
    image = inv_normalize(image)
    image = F.interpolate(
        image.unsqueeze(0),  # (1, 3, 224, 224)
        size=(upsample_size, upsample_size),  # New size
        mode='bilinear',
        align_corners=False
    ).squeeze(0)  # Back to (3, H, W)

    # Convert from (3, H, W) → (H, W, 3) for imshow
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    cmap = create_custom_cmap(n_slots)
    norm = plt.Normalize(vmin=0, vmax=n_slots-1)

    # Same for gt
    if gt is not None:
        gt = gt[batch_index].clone().detach().cpu()
        #print(gt.shape)

        gt = F.interpolate(
            gt.unsqueeze(0),  # (1, 3, 224, 224)
            size=(upsample_size, upsample_size),  # New size
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Back to (3, H, W)
        
        gt = gt.permute(1, 2, 0).detach().cpu().numpy()

    attn_slot_id = [np.argmax(attn_mask, axis=-1) for attn_mask in attn_masks_np]
    fused_mask = np.argmax(fused_mask, axis=-1)
    os.makedirs(os.path.join(base_path, str(iteration)), exist_ok=True)

    # Save every SA layer segmentation mask
    for i, attn in enumerate(attn_slot_id):
       # print(attn.shape)
        fig, ax = plt.subplots()
        ax.imshow(attn, cmap=cmap, norm = norm)
        ax.axis('off')
        output_path = os.path.join(base_path, str(iteration), f"sa_layer_{i}.png")
        #print(output_path)
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0)
        plt.close(fig) 
    
    # Save fused mask as well
    fig, ax = plt.subplots()
    ax.imshow(fused_mask, cmap=cmap, norm = norm)
    ax.axis('off')
    output_path = os.path.join(base_path, str(iteration), "fused_mask.png")
    plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0)
    plt.close(fig) 

    # Save image mask as well
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap, norm = norm)
    ax.axis('off')
    output_path = os.path.join(base_path, str(iteration), "image.png")
    plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0)
    plt.close(fig) 

    # Overlay
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=cmap, norm = norm)
    ax.imshow(fused_mask, cmap=cmap, norm = norm, alpha = 0.6)
    ax.axis('off')
    output_path = os.path.join(base_path, str(iteration), "image_mask_overlayed.png")
    plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0)
    plt.close(fig) 

    # Gt & gt overlay, if it is not none
    if gt is not None:
        fig, ax = plt.subplots()
        ax.imshow(image, cmap=cmap, norm = norm)
        ax.imshow(gt, cmap=cmap, norm = norm, alpha = 0.6)
        ax.axis('off')
        output_path = os.path.join(base_path, str(iteration), "image_gt_overlayed.png")
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0)
        plt.close(fig) 

        fig, ax = plt.subplots()
        ax.imshow(gt, cmap=cmap, norm = norm)
        ax.axis('off')
        output_path = os.path.join(base_path, str(iteration), "gt.png")
        plt.savefig(output_path, format="png", bbox_inches='tight', pad_inches=0)
        plt.close(fig) 

def visualize_layer_attn(attn_masks_list, image,  fused_mask, batch_index = 0, upsample_size=320, iteration=0, n_slots = 6, mode ="distinct", save_folder="spot/plots/default"):
    """
    attn_mask_list: torch.Size([b, 196, k]), n_scales
    """
    def create_custom_cmap(num_colors):
        if not (6 <= num_colors <= 24):
            raise ValueError("Number of colors must be between 6 and 24.")
        colors_6 = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b"   # Brown
        ]
        if num_colors == 6:
            return mcolors.ListedColormap(colors_6, name=f"custom_{num_colors}")
        colors_7 = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2"   # Pink
        ]
        if num_colors == 7:
            return mcolors.ListedColormap(colors_7, name=f"custom_{num_colors}")

        base_colors = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Lime
            "#17becf",  # Cyan
            "#aec7e8",  # Light Blue
            "#ffbb78",  # Light Orange
            "#98df8a",  # Light Green
            "#ff9896",  # Light Red
            "#c5b0d5",  # Light Purple
            "#c49c94",  # Light Brown
            "#f7b6d2",  # Light Pink
            "#c7c7c7",  # Light Gray
            "#dbdb8d",  # Light Lime
            "#9edae5",  # Light Cyan
            "#393b79",  # Dark Blue
            "#637939",  # Dark Green
            "#8c6d31",  # Dark Orange
            "#843c39"   # Dark Red
        ]

        colors = base_colors[:num_colors]
        return mcolors.ListedColormap(colors, name=f"custom_{num_colors}")
    
    
    # preprocess
    attn_mask_list = [attn_mask[batch_index] for attn_mask in attn_masks_list]
    h_w = int(math.sqrt(attn_mask_list[0].shape[0]))
    attn_mask_list_bi = [attn_mask.reshape(h_w, h_w, n_slots) for attn_mask in attn_mask_list]
    attn_mask_upsampled = [F.interpolate(
        attn_mask.permute(2, 0, 1).unsqueeze(0), 
        size=(upsample_size, upsample_size),  
        mode='bilinear'
    ).squeeze(0).permute(1, 2, 0)  for attn_mask in attn_mask_list_bi]
    attn_masks_np = [attn_mask.clone().detach().cpu().numpy() for attn_mask in attn_mask_upsampled]
    fused_mask = fused_mask[batch_index].reshape(h_w, h_w, n_slots)
    fused_mask = F.interpolate(
        fused_mask.permute(2, 0, 1).unsqueeze(0), 
        size=(320, 320),  
        mode='bilinear'
    ).squeeze(0).permute(1, 2, 0)
    fused_mask = fused_mask.clone().detach().cpu().numpy()
    image = image[batch_index].clone().detach().cpu()
    image = inv_normalize(image)
    image = image.permute(1, 2, 0)

    cmap = create_custom_cmap(n_slots)
    norm = plt.Normalize(vmin=0, vmax=n_slots-1)


    # Create subfolder for the current iteration
    iter_folder = os.path.join(save_folder, f"{iteration}")
    os.makedirs(iter_folder, exist_ok=True)
    
    attn_slot_id = [np.argmax(attn_mask, axis=-1) for attn_mask in attn_masks_np]
    fused_mask = np.argmax(fused_mask, axis=-1)
    
    # Save input image
    input_image_path = os.path.join(iter_folder, "input_image.png")
    plt.imsave(input_image_path, image)
    
    # Save each SA mask separately
    for i, attn in enumerate(attn_slot_id):
        sa_path = os.path.join(iter_folder, f"sa_layer_{i}.png")
        plt.imsave(sa_path, attn, cmap=cmap, norm=norm)
    
    # Save fused SA mask
    fused_mask_path = os.path.join(iter_folder, "fused_sa.png")
    plt.imsave(fused_mask_path, fused_mask, cmap=cmap, norm=norm)

    """
    if mode == "distinct":
   
        attn_slot_id = [np.argmax(attn_mask, axis=-1) for attn_mask in attn_masks_np]
        fused_mask = np.argmax(fused_mask, axis=-1)
        fig, axs = plt.subplots(1, len(attn_slot_id) +2, figsize=(24,3))
  
        axs[0].imshow(image)
        axs[0].axis('off')
        axs[0].set_title("Input image")
  
        for i, attn in enumerate(attn_slot_id):
            im = axs[i+1].imshow(attn, cmap=cmap, norm = norm)
            axs[i+1].set_title(f"SA at layer {i}")
            axs[i+1].axis('off') 
        axs[-1].imshow(fused_mask, cmap=cmap, norm = norm)
        axs[-1].set_title("Fused SA")
        axs[-1].axis('off')  
        fig.tight_layout()
        plt.savefig(f'{save_folder}/distinct_{iteration}.png')
        plt.close() 
    
      """
    #elif mode == 'overlay':
      #  fig, axs = plt.subplots(1, len(attn_masks_np) +2, figsize=(24,3))
      #  attn_masks_np.append(fused_mask)
      #  axs[0].imshow(image)
      #  axs[0].set_title("Input image")
      #  axs[0].axis('off')
      #  for i, attn in enumerate(attn_masks_np):
       #     slot_attn_masks = [attn[:, :, j] for j in range(attn.shape[2])] # 6x (320, 320)
       #     for slot_attn in slot_attn_masks:
        #        im = axs[i+1].imshow(slot_attn, cmap=cmap, alpha=0.3)
         #   axs[i+1].set_title(f"SA at layer {i}")
          #  axs[i+1].axis('off')  
        #axs[-1].set_title("Fused SA")

      #  fig.tight_layout()
       # plt.savefig(f'{save_folder}/overlay_{iteration}.png')
       #plt.close() 

        
    
        