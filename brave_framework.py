import torch
import random
import math
import torch
import torch.nn as nn
from transformers import BeitModel, ViTMAEModel, AutoImageProcessor, AutoProcessor, CLIPVisionModel
from typing import Any
from PIL import Image
from utils_spot import *
from slot_attn import BraveSlotAttentionEncoder
from transformer import TransformerDecoder
from mlp import MlpDecoder



class DinoEncoder():
    def __init__(self, args):
        self.encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16').eval()

        # All classes need these parameters set to initialize a corresponding slot attention module properly
        # Can be found out by just running an exemplary image, output is [batch_size, num_tokens, d_model]
        self.num_tokens = 196
        self.d_model = 768

        # Need to set this too: Does it need read tensors as input or image paths
        self.img_paths_as_input = False

        # Set requires grad to False for the encoder parameters
        for param_name, param in self.encoder.named_parameters():
            if ('blocks' in param_name):
                block_id = int(param_name.split('.')[1])
                if block_id >= args.finetune_blocks_after:
                    param.requires_grad = True  # update by gradient
                else:
                    param.requires_grad = False  # not update by gradient
            else:
                param.requires_grad = False  # not update by gradient
            
    
    def __call__(self, x):

        # Prepare input
        x = self.encoder.prepare_tokens(x)

        # encoder.blocks are ModuleList
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)

        # Remove the CLS
        return x[:, 1:]


class ClipEncoder():
    def __init__(self, args):

        # TOOD: Use clip-vit-large for better performance? check this
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
        self.encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")

        # All classes need these parameters set to initialize a corresponding slot attention module properly
        # Can be found out by just running an exemplary image, output is [batch_size, num_tokens, d_model]
        self.num_tokens = 196
        self.d_model = 768

        # Need to set this too: Does it need read tensors as input or image paths
        self.img_paths_as_input = True

        # Set requires grad to False for the encoder parameters
        for param_name, param in self.encoder.named_parameters():
            if ('blocks' in param_name):
                block_id = int(param_name.split('.')[1])
                if block_id >= args.finetune_blocks_after:
                    param.requires_grad = True  # update by gradient
                else:
                    param.requires_grad = False  # not update by gradient
            else:
                param.requires_grad = False  # not update by gradient

    def __call__(self, img_paths):
        
        # Preprocessing
        imgs = [Image.open(p) for p in img_paths]
        inputs = self.processor(images=imgs, return_tensors="pt")

        # Cast to CUDA
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        # Output has last_hidden_state (output of last layer) and hidden_sattes (Tuple of hidden states across all layers).
        # If needed for MSA, hidden_states can be used to access embedding across scales. Can also have attentions if
        # output_attentions=True is set.x
        outp = self.encoder(**inputs)

        # Delete to avoid OOM
        del imgs, inputs

        # Last hidden state has shape [batch, num_tokens + CLS, d_model], remove class token
        return outp.last_hidden_state[:, 1:]


class BeitEncoder():
    def __init__(self, args):

        self.processor = AutoImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k", use_fast=True)
        self.encoder = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")

        # All classes need these parameters set to initialize a corresponding slot attention module properly
        # Can be found out by just running an exemplary image, output is [batch_size, num_tokens, d_model]
        self.num_tokens = 196
        self.d_model = 768

        # Need to set this too: Does it need read tensors as input or image paths
        self.img_paths_as_input = True

        # Set requires grad to False for the encoder parameters
        for param_name, param in self.encoder.named_parameters():
            if ('blocks' in param_name):
                block_id = int(param_name.split('.')[1])
                if block_id >= args.finetune_blocks_after:
                    param.requires_grad = True  # update by gradient
                else:
                    param.requires_grad = False  # not update by gradient
            else:
                param.requires_grad = False  # not update by gradient

    def __call__(self, img_paths):
        # Preprocessing
        imgs = [Image.open(p) for p in img_paths]
        inputs = self.processor(images=imgs, return_tensors="pt")

        # Cast to CUDA
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        # Output has last_hidden_state (output of last layer) and hidden_sattes (Tuple of hidden states across all layers).
        # If needed for MSA, hidden_states can be used to access embedding across scales. Can also have attentions if
        # output_attentions=True is set.x
        outp = self.encoder(**inputs)

        # Delete to avoid OOM
        del imgs, inputs

        # Last hidden state has shape [batch, num_tokens + CLS, d_model], remove class token
        return outp.last_hidden_state[:, 1:]


class MAEEncoder():
    def __init__(self, args):
        self.processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base", use_fast=True)
        self.encoder = ViTMAEModel.from_pretrained("facebook/vit-mae-base")

        # Turn off masking! We are not fine-tuning anymore just inference, so we do NOT want the
        # model to perform masking anymore, otherwise the embedding size is reduced
        self.encoder.config.mask_ratio = 0.0

        # All classes need these parameters set to initialize a corresponding slot attention module properly
        # Can be found out by just running an exemplary image, output is [batch_size, num_tokens, d_model]
        self.num_tokens = 196
        self.d_model = 768

        # Need to set this too: Does it need read tensors as input or image paths
        self.img_paths_as_input = True

        # Set requires grad to False for the encoder parameters
        for param_name, param in self.encoder.named_parameters():
            if ('blocks' in param_name):
                block_id = int(param_name.split('.')[1])
                if block_id >= args.finetune_blocks_after:
                    param.requires_grad = True  # update by gradient
                else:
                    param.requires_grad = False  # not update by gradient
            else:
                param.requires_grad = False  # not update by gradient

    def __call__(self, img_paths):

        # Preprocessing
        imgs = [Image.open(p) for p in img_paths]
        inputs = self.processor(images=imgs, return_tensors="pt")

        # Cast to CUDA
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        # Output has last_hidden_state (output of last layer) and hidden_sattes (Tuple of hidden states across all layers).
        # If needed for MSA, hidden_states can be used to access embedding across scales. Can also have attentions if
        # output_attentions=True is set.x
        outp = self.encoder(**inputs)

        # Delete to avoid OOM
        del imgs, inputs

        # Last hidden state has shape [batch, num_tokens + CLS, d_model]
        # Remove class token
        return outp.last_hidden_state[:, 1:]


class BraveEncoder():
    def __init__(self, args):

        # Load in the different encoders. Make sure DinoEncoder is the last!
        self.encoder_suite = [
            ClipEncoder(args),
            MAEEncoder(args),
            DinoEncoder(args)
        ]   # TODO: Torch.nn.moduleList?

        assert all(
            [x.num_tokens == self.encoder_suite[-1].num_tokens for x in self.encoder_suite]
        ), "All encoders must have the same number of tokens to enable attention mask fusing"

    def __call__(self, images, image_paths):
        encoder_outputs = []
        for e in self.encoder_suite:
            encoder_outputs.append(e(image_paths if e.img_paths_as_input else images))
        return encoder_outputs

    def eval(self):
        for e in self.encoder_suite:
            e.encoder = e.encoder.eval()
        return self

    def cuda(self):
        for e in self.encoder_suite:
            e.encoder = e.encoder.cuda()
        return self


class BraveSPOT(nn.Module):
    def __init__(self, encoder, args, second_encoder=None):
        super().__init__()

        self.which_encoder = args.which_encoder
        self.encoder = encoder
        self.second_encoder = second_encoder
        self.encoder_final_norm = args.encoder_final_norm
        self.ms_which_encoder_layers = args.ms_which_encoder_layers            
        if self.second_encoder is not None:
            for param in self.second_encoder.parameters():
                param.requires_grad = False  # not update by gradient

        # Get number of tokens for images of size args.image_size and embedding size (d_model)
        num_tokens = self.encoder.encoder_suite[-1].num_tokens  # needed in the decoder, so should be of the model that provides the embedded target
        d_model = self.encoder.encoder_suite[-1].d_model

        args.d_model = d_model
        self.num_slots = args.num_slots
        self.d_model = args.d_model # input_channels
        
        self.slot_attn = BraveSlotAttentionEncoder(
            self.encoder, args.num_iterations, args.num_slots,
            args.slot_size, args.mlp_hidden_size, args.pos_channels,
            args.truncate, args.init_method, args.concat_method)

        self.input_proj = nn.Sequential(
            linear(args.d_model, args.d_model, bias=False),
            nn.LayerNorm(args.d_model),
        )
        
        size = int(math.sqrt(num_tokens))
        standard_order = torch.arange(size**2) # This is the default "left_top"
        
        self.cappa = args.cappa
        self.train_permutations = args.train_permutations
        
        if self.train_permutations == 'standard':
            self.permutations = [standard_order]
            self.eval_permutations = 'standard'
        
        else:
            standard_order_2d = standard_order.reshape(size,size)
            
            perm_top_left = torch.tensor([standard_order_2d[row,col] for col in range(0, size, 1) for row in range(0, size, 1)])
            
            perm_top_right = torch.tensor([standard_order_2d[row,col] for col in range(size-1, -1, -1) for row in range(0, size, 1)])
            perm_right_top = torch.tensor([standard_order_2d[row,col] for row in range(0, size, 1) for col in range(size-1, -1, -1)])
            
            perm_bottom_right = torch.tensor([standard_order_2d[row,col] for col in range(size-1, -1, -1) for row in range(size-1, -1, -1)])
            perm_right_bottom = torch.tensor([standard_order_2d[row,col] for row in range(size-1, -1, -1) for col in range(size-1, -1, -1)])
            
            perm_bottom_left = torch.tensor([standard_order_2d[row,col] for col in range(0, size, 1) for row in range(size-1, -1, -1)])
            perm_left_bottom = torch.tensor([standard_order_2d[row,col] for row in range(size-1, -1, -1) for col in range(0, size, 1)])
            
            perm_spiral = spiral_pattern(standard_order_2d, how = 'top_right')
            perm_spiral = torch.tensor((perm_spiral[::-1]).copy())
    
            self.permutations = [standard_order, # left_top
                                 perm_top_left, 
                                 perm_top_right, 
                                 perm_right_top, 
                                 perm_bottom_right, 
                                 perm_right_bottom,
                                 perm_bottom_left,
                                 perm_left_bottom,
                                 perm_spiral
                                 ]
            self.eval_permutations = args.eval_permutations

        self.perm_ind = list(range(len(self.permutations)))

        self.bos_tokens = nn.Parameter(torch.zeros(len(self.permutations), 1, 1, args.d_model))
        torch.nn.init.normal_(self.bos_tokens, std=.02)
        
        self.dec_type = args.dec_type
        self.use_slot_proj = args.use_slot_proj
        
        if self.dec_type=='mlp' and not self.use_slot_proj:
            self.slot_proj = nn.Identity()
            self.dec_input_dim = args.slot_size
        else:
            self.slot_proj = nn.Sequential(
                linear(args.slot_size, args.d_model, bias=False),
                nn.LayerNorm(args.d_model),
            )
            self.dec_input_dim = args.d_model
        
        if self.dec_type=='transformer':
            self.dec = TransformerDecoder(
                args.num_dec_blocks, args.max_tokens, args.d_model, args.num_heads, args.dropout, args.num_cross_heads)
            if self.cappa > 0:
                assert (self.train_permutations == 'standard') and (self.eval_permutations == 'standard')   
                self.mask_token = nn.Parameter(torch.zeros(1, 1, args.d_model))
                self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, args.d_model))
                torch.nn.init.normal_(self.pos_embed, std=.02)
                torch.nn.init.normal_(self.mask_token, std=.02)
                  
        elif self.dec_type=='mlp':
            self.dec = MlpDecoder(self.dec_input_dim, args.d_model, args.max_tokens, args.mlp_dec_hidden)

            assert (self.train_permutations == 'standard') and (self.eval_permutations == 'standard')  
        else:
            raise

        if self.dec_type=='transformer':
            # Register hook for capturing the cross-attention (of the query patch
            # tokens over the key/value slot tokens) from the last decoder
            # transformer block of the decoder.
            self.dec_slots_attns = []
            def hook_fn_forward_attn(module, input):
                self.dec_slots_attns.append(input[0])
            self.remove_handle = self.dec._modules["blocks"][-1]._modules["encoder_decoder_attn"]._modules["attn_dropout"].register_forward_pre_hook(hook_fn_forward_attn)


    def forward_encoder(self, x, encoder, image_paths):
        encoder.eval()

        return encoder(x, image_paths)

    
    def forward_decoder(self, slots, emb_target):
        # Prepate the input tokens for the decoder transformer:
        # (1) insert a learnable beggining-of-sequence ([BOS]) token at the beggining of each target embedding sequence.
        # (2) remove the last token of the target embedding sequence
        # (3) no need to add positional embeddings since positional information already exists at the DINO's outptu.
        

        if self.training:
            if self.train_permutations == 'standard':
                which_permutations = [0] # USE [0] FOR THE STANDARD ORDER
            elif self.train_permutations == 'random':
                which_permutations = [random.choice(self.perm_ind)]
            elif self.train_permutations == 'all':
                which_permutations = self.perm_ind
            else:
                raise
        else:
            if self.eval_permutations == 'standard':
                which_permutations = [0] # USE [0] FOR THE STANDARD ORDER
            elif self.eval_permutations == 'random':
                which_permutations = [random.choice(self.perm_ind)]
            elif self.eval_permutations == 'all':
                which_permutations = self.perm_ind
            else:
                raise
        
        
        all_dec_slots_attns = []
        all_dec_output = []
        
        for perm_id in which_permutations:
            current_perm = self.permutations[perm_id]

            bos_token = self.bos_tokens[perm_id]
            bos_token = bos_token.expand(emb_target.shape[0], -1, -1)
            
            use_pos_emb = self.cappa > 0
            parallel_dec = self.cappa > 0 and ((self.cappa >= 1.0) or (self.training and random.random() < self.cappa))
            #print(f"Paralled Decoder (CAPPA) {parallel_dec}")
            # Input to the decoder
            if parallel_dec: # Use parallel decoder
                dec_input = self.mask_token.to(emb_target.dtype).expand(emb_target.shape[0], -1, -1)
            else: # Use autoregressive decoder
                dec_input = torch.cat((bos_token, emb_target[:,current_perm,:][:, :-1, :]), dim=1)
      
            if use_pos_emb:
                # Add position embedding if they exist.
                dec_input = dec_input + self.pos_embed.to(emb_target.dtype)

            # dec_input has the same shape as emb_target, which is [B, N, D]
            dec_input = self.input_proj(dec_input)
    
            # Apply the decoder
            dec_input_slots = self.slot_proj(slots) # shape: [B, num_slots, D]
            if self.dec_type=='transformer':
                dec_output = self.dec(dec_input, dec_input_slots, causal_mask=(not parallel_dec))
                # decoder_output shape [B, N, D]

                dec_slots_attns = self.dec_slots_attns[0]
                self.dec_slots_attns = []

                # sum over the heads and 
                dec_slots_attns = dec_slots_attns.sum(dim=1) # [B, N, num_slots]
                # dec_slots_attns shape [B, num_heads, N, num_slots]
                # L1-normalize over the slots so as to sum to 1.
                dec_slots_attns = dec_slots_attns / dec_slots_attns.sum(dim=2, keepdim=True)

                inv_current_perm = torch.argsort(current_perm)
                dec_slots_attns = dec_slots_attns[:,inv_current_perm,:]
                dec_output = dec_output[:,inv_current_perm,:]

            elif self.dec_type=='mlp':
                dec_output, dec_slots_attns = self.dec(dec_input_slots)
                dec_slots_attns = dec_slots_attns.transpose(1,2)

            else:
                raise
            
            all_dec_slots_attns.append(dec_slots_attns)
            all_dec_output.append(dec_output)

        mean_dec_slots_attns = torch.stack(all_dec_slots_attns).mean(0)
        mean_dec_output = torch.stack(all_dec_output).mean(0)

        return mean_dec_output, mean_dec_slots_attns

    def get_embeddings_n_slots(self, image, image_paths):
        """
        image: batch_size x img_channels x H x W
        TODO: where is this called?
        """

        B, _, H, W = image.size()
        with torch.no_grad():
            emb_target = self.forward_encoder(image, self.encoder, image_paths) # List of ModuleList
        # emb_target shape: B, N, D

        # Apply the slot attention
        slots, slots_attns, _ = self.slot_attn(emb_target)
        return emb_target[-1], slots, slots_attns

    def forward(self, image, image_paths):
        """
        image: batch_size x img_channels x H x W
        """
        
        B, _, H, W = image.size()
        emb_input_lst = self.forward_encoder(image, self.encoder, image_paths)# forward encoder returns a list!
        with torch.no_grad():
            if self.second_encoder is not None:
                emb_target_lst = self.forward_encoder(image, self.second_encoder, image_paths) # TODO handle second encoder as multi-scale
            else:
                emb_target_lst = [emb_input.clone().detach() for emb_input in emb_input_lst]
        # emb_target shape: B, N, D ([64, 196, 768])
        emb_target = emb_target_lst[-1]
        
        # Apply the slot attention
        slots, slots_attns, init_slots, attn_logits = self.slot_attn(emb_target_lst)
        attn_logits = attn_logits.squeeze()
        # slots shape: [B, num_slots, Ds]
        # slots_attns shape: [B, N, num_slots]
        # slots_attns shape should be [64,196,6]
        # Apply the decoder.
        dec_recon, dec_slots_attns = self.forward_decoder(slots, emb_target)# TODO macht das Sinn?

        # Mean-Square-Error loss
        H_enc, W_enc = int(math.sqrt(emb_target.shape[1])), int(math.sqrt(emb_target.shape[1]))
        loss_mse = ((emb_target - dec_recon) ** 2).sum()/(B*H_enc*W_enc*self.d_model)

        # Reshape the slot and decoder-slot attentions.
        slots_attns = slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)
        dec_slots_attns = dec_slots_attns.transpose(-1, -2).reshape(B, self.num_slots, H_enc, W_enc)

        return loss_mse, slots_attns, dec_slots_attns, slots, dec_recon, attn_logits

