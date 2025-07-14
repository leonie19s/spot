import os

# Set available devices here, do NOT use GPU 0 on node 20
device_ids =[0]
os.environ["CUDA_VISIBLE_DEVICES"]=", ".join(str(device_id) for device_id in device_ids)

VOC_SPLIT = "many"

import copy
import os.path
import argparse
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision.utils import save_image
from spot import SPOT
from ms_spot import MSSPOT
from datasets import PascalVOC
from ocl_metrics import UnsupervisedMaskIoUMetric, ARIMetric
from utils_spot import inv_normalize, visualize, bool_flag, reduce_dataset
import models_vit
import matplotlib.pyplot as plt


def plotting_later_analysis():

    single = [0.6147210597991943, 0.2322993278503418, 0.08499794453382492, 0.06798166781663895]
    few = [0.6353044509887695, 0.22225765883922577, 0.08019126951694489, 0.06224660947918892]
    some = [0.6485604643821716, 0.21646356582641602, 0.07689111679792404, 0.058084893971681595]
    many = [0.6499180197715759, 0.21554143726825714, 0.0779801681637764, 0.0565604530274868]
    concat = [single, few, some, many]

    layer0 = [x[0] for x in concat]
    layer1 = [x[1] for x in concat]
    layer2 = [x[2] for x in concat]
    layer3 = [x[3] for x in concat]
    all_other_layers = [x[1] + x[2] + x[3] for x in concat]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    axes[0].plot(layer0, label="Earliest layer")
    axes[1].plot(all_other_layers, label="Layer laters")
    axes[0].set_xticks(ticks=[0, 1, 2, 3], labels=["x = 1", "2 <= x <= 3", "4 <= x <= 6", "x >= 7"])
    axes[1].set_xticks(ticks=[0, 1, 2, 3], labels=["x = 1", "2 <= x <= 3", "4 <= x <= 6", "x >= 7"])
    axes[0].set_xlabel("Number of objects in scene")
    axes[0].set_ylabel("Mean gated weights")
    axes[1].set_xlabel("Number of objects in scene")
    axes[0].set_title("Layer 9")
    axes[1].set_title("Layer 10+11+12")
    plt.savefig("iccv_logs/voc_obj_count_eval/early_vs_later_layers.png")

    """
    plt.plot(layer0, label="Layer 9")
    plt.plot(layer1, label="Layer 10")
    plt.plot(layer2, label="Layer 11")
    plt.plot(layer3, label="Layer 12")
    plt.xticks(ticks=[0, 1, 2, 3], labels=["x = 1", "2 <= x <= 3", "4 <= x <= 6", "x >= 7"])

    plt.xlabel("Number of objects in scene")
    plt.ylabel("Mean gated weights per layer")
    plt.savefig("iccv_logs/voc_obj_count_eval/alllayersinone.png")
    """

    layers = [layer0, layer1, layer2, layer3]
    labels = ["Layer 9", "Layer 10", "Layer 11", "Layer 12"]
    xtick_positions = [0, 1, 2, 3]
    xtick_labels = ["x = 1", "2 <= x <= 3", "4 <= x <= 6", "x >= 7"]

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), sharex=True)
    axes = axes.flatten()

    for i in range(4):
        ax = axes[i]
        ax.plot(layers[i])
        ax.set_title(labels[i])
        ax.set_ylabel("Mean gated weights")
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels)

    # Label only bottom row x-axis
    for ax in axes[2:]:
        ax.set_xlabel("Number of objects in scene")

    # fig.suptitle("Layer-wise Gated Weight Analysis", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig("iccv_logs/voc_obj_count_eval/layerwise.png")

""" 
    computes the BF (Boundary F1) contour matching score between the predicted and GT segmentation 
    https://github.com/minar09/bfscore_python/blob/master/bfscore.py
"""

import numpy as np
import cv2
major = cv2.__version__.split('.')[0]     # Get opencv version

def calc_precision_recall(contours_a, contours_b, threshold):
    x = contours_a
    y = contours_b

    xx = np.array(x)
    hits = []
    for yrec in y:
        d = np.square(xx[:,0] - yrec[0]) + np.square(xx[:,1] - yrec[1])
        hits.append(np.any(d < threshold*threshold))
    top_count = np.sum(hits)

    try:
        precision_recall = top_count / len(y)
    except ZeroDivisionError:
        precision_recall = 0

    return precision_recall, top_count, len(y)

def bfscore(gtfile, prfile, threshold=2):

    gt__ = cv2.imread(gtfile)    # Read GT segmentation
    gt_ = cv2.cvtColor(gt__, cv2.COLOR_BGR2GRAY)    # Convert color space

    pr_ = cv2.imread(prfile)    # Read predicted segmentation
    pr_ = cv2.cvtColor(pr_, cv2.COLOR_BGR2GRAY)    # Convert color space

    classes_gt = np.unique(gt_)    # Get GT classes
    classes_pr = np.unique(pr_)    # Get predicted classes

    # Check classes from GT and prediction
    if not np.array_equiv(classes_gt, classes_pr):
        print('Classes are not same! GT:', classes_gt, 'Pred:', classes_pr)

        classes = np.concatenate((classes_gt, classes_pr))
        classes = np.unique(classes)
        classes = np.sort(classes)
        print('Merged classes :', classes)
    else:
        print('Classes :', classes_gt)
        classes = classes_gt    # Get matched classes

    m = np.max(classes)    # Get max of classes (number of classes)
    # Define bfscore variable (initialized with zeros)
    bfscores = np.zeros((m+1), dtype=float)
    areas_gt = np.zeros((m + 1), dtype=float)

    for i in range(m+1):
        bfscores[i] = np.nan
        areas_gt[i] = np.nan

    for target_class in classes:    # Iterate over classes

        if target_class == 0:     # Skip background
            continue

        gt = gt_.copy()
        gt[gt != target_class] = 0
        # print(gt.shape)

        # contours는 point의 list형태.
        if major == '3':    # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)    # Find contours of the shape
        else:    # For other opencv versions
            contours, _ = cv2.findContours(
                gt, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # Find contours of the shape

        # contours 는 list of numpy arrays
        contours_gt = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_gt.append(contours[i][j][0].tolist())

        # Get contour area of GT
        if contours_gt:
            area = cv2.contourArea(np.array(contours_gt))
            areas_gt[target_class] = area

        pr = pr_.copy()
        pr[pr != target_class] = 0
        # print(pr.shape)

        # contours는 point의 list형태.
        if major == '3':    # For opencv version 3.x
            _, contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        else:    # For other opencv versions
            contours, _ = cv2.findContours(
                pr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # contours 는 list of numpy arrays
        contours_pr = []
        for i in range(len(contours)):
            for j in range(len(contours[i])):
                contours_pr.append(contours[i][j][0].tolist())

        # 3. calculate
        precision, numerator, denominator = calc_precision_recall(
            contours_gt, contours_pr, threshold)    # Precision
        print("\tprecision:", denominator, numerator)

        recall, numerator, denominator = calc_precision_recall(
            contours_pr, contours_gt, threshold)    # Recall
        print("\trecall:", denominator, numerator)

        f1 = 0
        try:
            f1 = 2*recall*precision/(recall + precision)    # F1 score
        except:
            #f1 = 0
            f1 = np.nan
        print("\tf1:", f1)
        bfscores[target_class] = f1

    # return bfscores[1:], np.sum(bfscores[1:])/len(classes[1:])    # Return bfscores, except for background, and per image score
    return bfscores[1:], areas_gt[1:]    # Return bfscores, except for background

class FusionModuleHook:
    def __init__(self, module):
        self.inputs = None
        self.outputs = None
        self.hook = module.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        # input is a tuple; output can be tuple or tensor
        self.inputs = input
        self.outputs = output

    def clear(self):
        self.inputs = None
        self.outputs = None

    def remove(self):
        self.hook.remove()

    def detach(self, x):
        if isinstance(x, (list, tuple)):
            return [self.detach(t) for t in x]
        elif isinstance(x, torch.Tensor):
            return x.detach().cpu()
        else:
            return x
            
    def __call__(self):

        # Detach all tensors
        detached_inputs = self.detach(self.inputs)
        detached_outputs = self.detach(self.outputs)

        # Get input and outputs of fusion module
        slots_list, attn_list, init_slots_list, attn_logits_list = detached_inputs
        agg_slots, agg_attn, agg_init_slots, agg_attn_logits = detached_outputs

        # Clear internal storage to avoid memory issues
        self.clear()

        # Return whatever necessary
        return slots_list, agg_slots


parser = argparse.ArgumentParser()

parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--val_image_size', type=int, default=224)
parser.add_argument('--val_mask_size', type=int, default=320)
parser.add_argument('--eval_batch_size', type=int, default=32)
parser.add_argument('--viz_resolution_factor', type=float, default=0.5)

parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
parser.add_argument('--log_path', default='results')
parser.add_argument('--dataset', default='coco', help='coco or voc')
parser.add_argument('--data_path',  type=str, help='dataset path')

parser.add_argument('--num_dec_blocks', type=int, default=4)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--num_heads', type=int, default=6)
parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--num_iterations', type=int, default=3)
parser.add_argument('--num_slots', type=int, default=7)
parser.add_argument('--slot_size', type=int, default=256)
parser.add_argument('--mlp_hidden_size', type=int, default=1024)
parser.add_argument('--img_channels', type=int, default=3)
parser.add_argument('--pos_channels', type=int, default=4)
parser.add_argument('--num_cross_heads', type=int, default=None)

parser.add_argument('--dec_type',  type=str, default='transformer', help='type of decoder transformer or mlp')
parser.add_argument('--cappa', type=float, default=-1)
parser.add_argument('--mlp_dec_hidden',  type=int, default=2048, help='Dimension of decoder mlp hidden layers')
parser.add_argument('--use_slot_proj',  type=bool_flag, default=True, help='Use an extra projection before MLP decoder')

parser.add_argument('--which_encoder',  type=str, default='dino_vitb16', help='dino_vitb16, dino_vits8, dinov2_vitb14_reg, dinov2_vits14_reg, dinov2_vitb14, dinov2_vits14, mae_vitb16')
parser.add_argument('--finetune_blocks_after',  type=int, default=100, help='just use a large number')
parser.add_argument('--encoder_final_norm',  type=bool_flag, default=False)

parser.add_argument('--truncate',  type=str, default='bi-level', help='bi-level or fixed-point or none')
parser.add_argument('--init_method', default='embedding', help='embedding or shared_gaussian')

parser.add_argument('--use_second_encoder',  type= bool_flag, default = True, help='different encoder for input and target of decoder')

parser.add_argument('--train_permutations',  type=str, default='random', help='it is just for the initialization')
parser.add_argument('--eval_permutations',  type=str, default='standard', help='standard, random, or all')

parser.add_argument('--ms_which_encoder_layers', type=str, default="9,10,11", help= "Which block layers of the encoders are to be used for multi-scale slot attention, values as ints separated by commas with no whitespace")
parser.add_argument('--concat_method', type=str, default='mean', help="how the multiscale attention is concatenated, choose from ['mean', 'sum', 'residual, 'max', 'denseconnector', 'transformerconnector']")
parser.add_argument('--shared_weights', type=bool, default=False, help='if the weights of the slot attention encoder module are shared')
parser.add_argument('--data_cut', type=float, default=1, help='factor how much of the original length of the data is used')
parser.add_argument('--log_folder_name', type=str, default=None, help='folder to save the logs and model')
parser.add_argument('--visualize_attn', type=bool, default=False)  

args = parser.parse_args()

torch.manual_seed(args.seed)

# Directly transform string of layers into proper list
args_layers_list = list(map(int, args.ms_which_encoder_layers.split(',')))
assert len(args_layers_list) > 0, "ms_which_encoder_layers must contain at least one integer"
assert all(isinstance(x, int) for x in args_layers_list), "ms_which_encoder_layers must contain only integers, separated by commas"
args.ms_which_encoder_layers = args_layers_list

arg_str_list = ['{}={}'.format(k, v) for k, v in vars(args).items()]
arg_str = '__'.join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat()) if args.log_folder_name is None else os.path.join(args.log_path, args.log_folder_name)
args.log_dir = log_dir

val_dataset = PascalVOC(root=args.data_path, split=VOC_SPLIT, image_size=args.val_image_size, mask_size = args.val_mask_size)

args.max_tokens = int((args.val_image_size/16)**2)

val_sampler = None

loader_kwargs = {
    'num_workers': args.num_workers,
    'pin_memory': True,
}

val_loader = DataLoader(val_dataset, sampler=val_sampler, shuffle=False, drop_last = False, batch_size=args.eval_batch_size, **loader_kwargs)

val_epoch_size = len(val_loader)

if args.which_encoder == 'dino_vitb16':
    args.max_tokens = int((args.val_image_size/16)**2)
    encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
elif args.which_encoder == 'dino_vits8':
    args.max_tokens = int((args.val_image_size/8)**2)
    encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
elif args.which_encoder == 'dino_vitb8':
    args.max_tokens = int((args.val_image_size/8)**2)
    encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')
elif args.which_encoder == 'dinov2_vitb14':
    args.max_tokens = int((args.val_image_size/14)**2)
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
elif args.which_encoder == 'dinov2_vits14':
    args.max_tokens = int((args.val_image_size/14)**2)
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
elif args.which_encoder == 'dinov2_vitb14_reg':
    args.max_tokens = int((args.val_image_size/14)**2)
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
elif args.which_encoder == 'dinov2_vits14_reg':
    args.max_tokens = int((args.val_image_size/14)**2)
    encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
elif args.which_encoder == 'mae_vitb16':
    args.max_tokens = int((args.val_image_size/16)**2)
    encoder = models_vit.__dict__["vit_base_patch16"](num_classes=0, global_pool=False, drop_path_rate=0)
else:
    raise
        
encoder = encoder.eval()

if args.use_second_encoder:
    encoder_second = copy.deepcopy(encoder).eval()
else:
    encoder_second = None

if args.num_cross_heads is None:
    args.num_cross_heads = args.num_heads


model = MSSPOT(encoder, args, encoder_second)
# model = SPOT(encoder, args, encoder_second)

checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
checkpoint['model'] = {k.replace("tf_dec.", "dec."): v for k, v in checkpoint['model'].items()} # compatibility with older runs
model.load_state_dict(checkpoint['model'], strict = True)

model = model.cuda()

# Register hooks to the forward pass of the slot attention module
fusion_hook = FusionModuleHook(model.slot_attn.fusion_module)

MBO_c_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
MBO_i_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
miou_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).cuda()
ari_metric = ARIMetric(foreground = True, ignore_overlaps = True).cuda()

MBO_c_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
MBO_i_slot_metric = UnsupervisedMaskIoUMetric(matching="best_overlap", ignore_background = True, ignore_overlaps = True).cuda()
miou_slot_metric = UnsupervisedMaskIoUMetric(matching="hungarian", ignore_background = True, ignore_overlaps = True).cuda()
ari_slot_metric = ARIMetric(foreground = True, ignore_overlaps = True).cuda()

with torch.no_grad():
    model.eval()

    val_mse = 0.
    counter = 0

    acc = []

    for batch, (image, true_mask_i, true_mask_c, mask_ignore) in enumerate(tqdm(val_loader)):
        image = image.cuda()
        true_mask_i = true_mask_i.cuda()
        true_mask_c = true_mask_c.cuda()
        mask_ignore = mask_ignore.cuda() 
        
        batch_size = image.shape[0]
        counter += batch_size

        mse, default_slots_attns, dec_slots_attns, _, _, _, _, _ = model(image)

        # Retrieve slots and fused slots from hook
        slot_list, fused_slots = fusion_hook()

        # Compute contribution of each individual slot of each layer to the fused slots through cosine similarity along feature dimension
        contribution_weights = torch.stack([F.cosine_similarity(fused_slots, slot, dim=-1) for slot in slot_list], dim=-1)  # Shape: [B, num_slots, num_layers]
        contribution_weights = F.softmax(contribution_weights, dim=-1)
        acc.append(contribution_weights.detach().cpu())

        # DINOSAUR uses as attention masks the attenton maps of the decoder
        # over the slots, which bilinearly resizes to match the image resolution
        # dec_slots_attns shape: [B, num_slots, H_enc, W_enc]
        default_attns = F.interpolate(default_slots_attns, size=args.val_mask_size, mode='bilinear')
        dec_attns = F.interpolate(dec_slots_attns, size=args.val_mask_size, mode='bilinear')
        # dec_attns shape [B, num_slots, H, W]
        default_attns = default_attns.unsqueeze(2)
        dec_attns = dec_attns.unsqueeze(2) # shape [B, num_slots, 1, H, W]

        pred_default_mask = default_attns.argmax(1).squeeze(1)
        pred_dec_mask = dec_attns.argmax(1).squeeze(1)

        val_mse += mse.item()
             
        # Compute ARI, MBO_i and MBO_c, miou scores for both slot attention and decoder
        true_mask_i_reshaped = torch.nn.functional.one_hot(true_mask_i).to(torch.float32).permute(0,3,1,2).cuda()
        true_mask_c_reshaped = torch.nn.functional.one_hot(true_mask_c).to(torch.float32).permute(0,3,1,2).cuda()
        pred_dec_mask_reshaped = torch.nn.functional.one_hot(pred_dec_mask).to(torch.float32).permute(0,3,1,2).cuda()
        pred_default_mask_reshaped = torch.nn.functional.one_hot(pred_default_mask).to(torch.float32).permute(0,3,1,2).cuda()
        
        MBO_i_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
        MBO_c_metric.update(pred_dec_mask_reshaped, true_mask_c_reshaped, mask_ignore)
        miou_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
        ari_metric.update(pred_dec_mask_reshaped, true_mask_i_reshaped, mask_ignore)
    
        MBO_i_slot_metric.update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
        MBO_c_slot_metric.update(pred_default_mask_reshaped, true_mask_c_reshaped, mask_ignore)
        miou_slot_metric.update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)
        ari_slot_metric.update(pred_default_mask_reshaped, true_mask_i_reshaped, mask_ignore)

    val_mse /= (val_epoch_size)
    ari = 100 * ari_metric.compute()
    ari_slot = 100 * ari_slot_metric.compute()
    mbo_c = 100 * MBO_c_metric.compute()
    mbo_i = 100 * MBO_i_metric.compute()
    miou = 100 * miou_metric.compute()
    mbo_c_slot = 100 * MBO_c_slot_metric.compute()
    mbo_i_slot = 100 * MBO_i_slot_metric.compute()
    miou_slot = 100 * miou_slot_metric.compute()
    val_loss = val_mse

    df_results = pd.DataFrame([[mbo_i.item(), mbo_c.item(), ari.item(),  val_mse, mbo_i_slot.item(), mbo_c_slot.item(), ari_slot.item(), miou.item(), miou_slot.item()]], 
                 columns=['mBO_i', 'mBO_c', 'FG-ARI',  'MSE', 'mBO_i_slots', 'mBO_c_slots', 'FG-ARI_slots', 'miou', 'miou_slots'])

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_results.to_string())

    # Print final layer-wise mean contributions
    all_contributions = torch.cat(acc, dim=0)  # [samples, S, L]
    mean_contributions_per_layer = all_contributions.mean(dim=(0, 1))
    print("Mean contributions per layer to fused slot representation")
    print(mean_contributions_per_layer)
        
    if args.concat_method == "gatedfusion":
        print(f"==> Gated weights mean: {model.slot_attn.fusion_module.get_mean_gates()}")
    
    plotting_later_analysis()

