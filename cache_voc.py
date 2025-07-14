import os

device_ids =[2]
os.environ["CUDA_VISIBLE_DEVICES"]=", ".join(str(device_id) for device_id in device_ids)
from tqdm import tqdm
import torch
from datasets import PascalVOC

split = "val"
ROOT = f"/fastdata/vilab01/voc_cache_{split}"
os.makedirs(ROOT, exist_ok=True) # create folder once
LAYERS = [8,9,10,11]

def forward_encoder(x, encoder, which_encoder='dino_vitb16', encoder_final_norm=False, ms_which_encoder_layers=[8,9,10,11]):
        encoder.eval()
        if which_encoder in ['dinov2_vitb14', 'dinov2_vits14', 'dinov2_vitb14_reg', 'dinov2_vits14_reg']:
            x = encoder.prepare_tokens_with_masks(x, None)
        else:
            x = encoder.prepare_tokens(x)
        ms_x_temp = []
        ms_x =[]
        for i, blk in enumerate(encoder.blocks):
            x = blk(x)
            if i == len(encoder.blocks) - 1 and encoder_final_norm:
                x = encoder.norm(x)
            if i in ms_which_encoder_layers:
                ms_x_temp.append(x)
        offset = 1
        if which_encoder in ['dinov2_vitb14_reg', 'dinov2_vits14_reg']:
            offset += encoder.num_register_tokens
        elif which_encoder in ['simpool_vits16']:
            offset += -1

        for x in ms_x_temp:
            x = x[:, offset :] # remove the [CLS] and (if they exist) registers tokens 
            ms_x.append(x)
        
        return ms_x


def save_encoded(file_name, encoded_layers, output_dir = ROOT, which_layers = LAYERS):
    image_folder = os.path.join(output_dir, file_name)
    os.makedirs(image_folder, exist_ok=True)
    for i, layer in zip(which_layers, encoded_layers): # iterate over layers
        save_path = os.path.join(output_dir, file_name, f"layer_{i}")
        os.makedirs(save_path, exist_ok=True) # create layer subfolder
        out_file = os.path.join(save_path, "cache.pt")
        if not os.path.exists(out_file): # skip existing
            torch.save(layer.cpu(), out_file)

def main():
    encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').eval().cuda()
    config = {
        "trainaug":
        {
            "image_size": 224,
            "mask_size": 224
        },
        "val":
        {
            "image_size" : 224,
            "mask_size" : 320
        }
    }
    dataset = PascalVOC(root="/fastdata/vilab01/VOCdevkit/VOC2012", split=split, image_size=config[split]['image_size'], mask_size=config[split]['mask_size'])
    for index in tqdm(range(len(dataset)), desc="encoding voc"):

        if split == "trainaug":
            image, file_name = dataset[index]
        else:
            image, _, _, _, file_name = dataset[index]
        
        # skip if already encoded
        test_cache_path = os.path.join(ROOT, file_name, f"layer_{LAYERS[0]}", "cache.pt")
        if os.path.exists(test_cache_path):
            continue
        
        with torch.no_grad():
            encoded_layers = forward_encoder(image.unsqueeze(0).cuda(), encoder)  # add batch dimension
        save_encoded(file_name, encoded_layers)


if __name__ == "__main__":
    main()