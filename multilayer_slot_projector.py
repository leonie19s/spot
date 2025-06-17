import torch
import torch.nn as nn
import torch.nn.functional as F


class MapFusionSimple():
    """
        Module for fusing the attention maps, init slots and logits together through a simple aggregation
        function, such as mean, max or sum.
    """

    def __init__(self, num_image_patches, num_layers, fct=None):
        self.fct = fct

    def __call__(self, fused_slots, slot_list, slot_att_list, init_slot_list, attn_logits_list):
        """
            fused_slots: Final fused slot [B, num_slots, slot_dim]
            slot_list: List of slots [B, num_slots, slot_dim], e.g. VOC [64, 6, 256] with len = num_layers
            slot_att_list: List of slot attention maps [B, H_emb * W_emb, num_slots], e.g. VOC [64, 196, 6] with len = num_layers
            init_slot_list: List of slot initializations [B, num_slots, slot_dim] with len = num_layers
            attn_logits_list: List of attention logits [B, 1, H_emb * W_emb, num_slots]) with len = num_layers
        """
        fused_attn = self.fct(torch.stack(slot_att_list), dim=0)
        fused_init_slots = self.fct(torch.stack(init_slot_list), dim=0)
        fused_attn_logits = self.fct(torch.stack(attn_logits_list), dim=0)
        return fused_attn, fused_init_slots, fused_attn_logits


class MapFusionWithLearnedWeights(nn.Module):
    """
        Module for fusing the attention maps, init slots and logits together through a linear combination
        with learned weights that sum to 1.
    """
    
    def __init__(self, num_image_patches, num_layers):
        super(MapFusionWithLearnedWeights, self).__init__()
        
        # Learnable weights for each layer, initialize uniformly at the beginning
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)

    def forward(self, fused_slots, slot_list, slot_att_list, init_slot_list, attn_logits_list):
        """
            fused_slots: Final fused slot [B, num_slots, slot_dim]
            slot_list: List of slots [B, num_slots, slot_dim], e.g. VOC [64, 6, 256] with len = num_layers
            slot_att_list: List of slot attention maps [B, H_emb * W_emb, num_slots], e.g. VOC [64, 196, 6] with len = num_layers
            init_slot_list: List of slot initializations [B, num_slots, slot_dim] with len = num_layers
            attn_logits_list: List of attention logits [B, 1, H_emb * W_emb, num_slots]) with len = num_layers
        """

        # Stack attention maps along new dimension for layers: [B, H*W, num_slots, num_layers]
        stacked_maps = torch.stack(slot_att_list, dim=3) 
        stacked_init_slots = torch.stack(init_slot_list, dim=3)
        stacked_attn_logits = torch.stack(attn_logits_list, dim=4)
        
        # Normalize so weights sum to 1
        normalized_weights = torch.softmax(self.layer_weights, dim=0)
        #print(normalized_weights)
        # Apply weights to attention maps: [B, H*W, num_slots, num_layers]
        weighted_maps = stacked_maps * normalized_weights.view(1, 1, 1, -1)
        weighted_init_slots = stacked_init_slots * normalized_weights.view(1, 1, 1, -1)
        weighted_attn_logits = stacked_attn_logits * normalized_weights.view(1, 1, 1, 1, -1)
        
        # Sum across the layer dimension to fuse them together: [B, H*W, num_slots]
        fused_maps = weighted_maps.sum(dim=-1)
        fused_init_slots = weighted_init_slots.sum(dim=-1)
        fused_attn_logits = weighted_attn_logits.sum(dim=-1)
        
        return fused_maps, fused_init_slots, fused_attn_logits
    

class MapFusionPixelwiseWithLearnedWeights(nn.Module):
    """
        Module for fusing the attention maps, init slots and logits together through a linear combination
        with learned weights. Each pixel of the embedding (H_emb, W_emb) has its own weight that is
        learned to sum to 1 across layers.
    """

    def __init__(self, num_image_patches, num_layers):
        super(MapFusionPixelwiseWithLearnedWeights, self).__init__()
        
        # Learnable weights for each pixel and each layer
        self.num_image_patches = num_image_patches
        self.pixel_weights = nn.Parameter(torch.randn(num_image_patches, num_layers))  # [H*W, num_layers]
    
    def forward(self, fused_slots, slot_list, slot_att_list, init_slot_list, attn_logits_list):
        """
            Fuse multi-layer attention maps pixel-wise into one map per slot.

            fused_slots: Final fused slot [B, num_slots, slot_dim]
            slot_list: List of slots [B, num_slots, slot_dim], e.g. VOC [64, 6, 256] with len = num_layers
            slot_att_list: List of slot attention maps [B, H_emb * W_emb, num_slots], e.g. VOC [64, 196, 6] with len = num_layers
            init_slot_list: List of slot initializations [B, num_slots, slot_dim] with len = num_layers
            attn_logits_list: List of attention logits [B, 1, H_emb * W_emb, num_slots]) with len = num_layers
        """

        # Sanity check
        if self.num_image_patches != slot_att_list[0].shape[1]:
            raise ValueError("Embedding size H_emb * W_emb specified in MapFusionPixelwiseWithLearnedWeights does not match input!")
        
        # Stack attention maps along the layer dimension at the end: [B, H*W, num_slots, num_layers]
        stacked_maps = torch.stack(slot_att_list, dim=-1) 

        # Reshape pixel weights for applying to stacked map to [1, H*W, 1, num_layers]
        normalized_weights = torch.softmax(self.pixel_weights, dim=1).view(1, -1, 1, stacked_maps.shape[-1])
        
        # Apply weights to attention maps [B, H*W, num_slots, num_layers]
        weighted_maps = stacked_maps * normalized_weights

        # Sum across the layer dimension to fuse across layers [B, H*W, num_slots]
        fused_maps = weighted_maps.sum(dim=-1)
        
        # Take the mean for the logits and slot_inits, because we cannot weigh them pixel-wise (logits theoretically possible)
        fused_inits = torch.mean(torch.stack(init_slot_list), dim=0)
        fused_att_logits = torch.mean(torch.stack(attn_logits_list), dim=0)

        return fused_maps, fused_inits, fused_att_logits


class MapFusionThroughContribution():
    """
        Module for fusing the attention maps, init slots and logits together through a linear combination
        with weights based on the individual contribution of each slot attention layer to the final
        fused slot representation.
    """

    def __init__(self, num_image_patches, num_layers):
        self.num_layers = num_layers
            
    def __call__(self, fused_slots, slot_list, slot_att_list, init_slot_list, attn_logits_list):
        """
            fused_slots: Final fused slot [B, num_slots, slot_dim]
            slot_list: List of slots [B, num_slots, slot_dim], e.g. VOC [64, 6, 256] with len = num_layers
            slot_att_list: List of slot attention maps [B, H_emb * W_emb, num_slots], e.g. VOC [64, 196, 6] with len = num_layers
            init_slot_list: List of slot initializations [B, num_slots, slot_dim] with len = num_layers
            attn_logits_list: List of attention logits [B, 1, H_emb * W_emb, num_slots]) with len = num_layers
        """

        # Compute contribution of each individual slot of each layer to the fused slots through cosine similarity along feature dimension
        weights = torch.stack([F.cosine_similarity(fused_slots, slot, dim=-1) for slot in slot_list], dim=-1)  # Shape: [B, num_slots, num_layers]
        
        # Normalize contributions
        weights = F.softmax(weights, dim=-1)

        # Fuse the attention maps based on the individual contributions
        fused_attn = torch.zeros_like(slot_att_list[0]) # [B, H_emb * W_emb, num_slots]
        for l_idx, attn in enumerate(slot_att_list):
            fused_attn += weights[:, :, l_idx].unsqueeze(1) * attn

        # Fuse init slots through contributions as well
        fused_inits = torch.zeros_like(slot_list[0]) # [B, num_slots, slot_dim]
        for l_idx, init_slots in enumerate(init_slot_list):
            fused_inits += weights[:, :, l_idx].unsqueeze(2) * init_slots

        # Fuse slot logits through contributions as well
        fused_logits = torch.zeros_like(attn_logits_list[0]) # [B, 1, H_emb * W_emb, num_slots]
        for l_idx, attn_logits in enumerate(attn_logits_list):
            fused_logits += weights[:, :, l_idx].unsqueeze(1).unsqueeze(1) * attn_logits

        return fused_attn, fused_inits, fused_logits 


class SimpleConnector(nn.Module):
    """
        Wrapper class for simple fusion of slot attention outputs.
    """
    def __init__(self, slot_dim, num_layers, fct=None):
        super(SimpleConnector, self).__init__()
        self.fct = fct
        self.map_fuser = MapFusionSimple(196, num_layers, fct)
        
    def __call__(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):
        fused_slots = self.fct(torch.stack(slot_list), dim=0)
        fused_attn, fused_inits, fused_logits = self.map_fuser(fused_slots, slot_list, slot_att_list, init_slot_list, attn_logits_list)
        return fused_slots, fused_attn, fused_inits, fused_logits


class NormWeightConnector():
    def __init__(self, slot_dim, num_layers):
        pass

    def __call__(self, slots_tensor_list, attn_tensor_list, init_slots_list, attn_logits_list):
        """ 
            shape of slots list: [torch.Size([b, k, f])] *  s
            shape of attn list: [torch.Size([b, l, k])] * s
            shape of init slots list: [torch.Size([b, k, f])]* s
            shape of attn logits list: [torch.Size([b, i, l, k])]* s (i=1)
        """
        sum_attention_masks = torch.sum(torch.stack(attn_tensor_list, dim=0), dim=0)  # torch.Size([64, 196, 6])
        scale_contributions = [torch.sum(mask, dim=1, keepdim=True) for mask in attn_tensor_list]  # torch.Size([64, 1, 6]) x 4
        scale_weights = [contribution / torch.sum(sum_attention_masks, dim=1, keepdim=True) for contribution in scale_contributions] # torch.Size([64, 1, 6])

        combined_slots = torch.stack(slots_tensor_list, dim=0)  # torch.Size([4, 64, 6, 256])
        scale_weights_slots = torch.stack(scale_weights, dim=0).squeeze(2)  # torch.Size([4, 64, 6])
        fused_slots = torch.einsum('sbk,sbkf->bkf', scale_weights_slots, combined_slots)

        combined_attention_masks = torch.stack(attn_tensor_list, dim=0)  # torch.Size([4, 64, 196, 6]) sblk
        scale_weights_masks = torch.stack(scale_weights, dim=0)  # torch.Size([4, 64, 1, 6]) sbik
        fused_attention_masks = torch.einsum('sblk,sbik->bik', scale_weights_masks, combined_attention_masks)  # [B, N, K]
        
        combined_init_slots = torch.stack(init_slots_list, dim=0)  # torch.Size([4, 64, 6, 256]) sbkf
        fused_init_slots = torch.einsum('sbk,sbkf->bkf', scale_weights_slots, combined_init_slots)  #torch.Size([64, 6, 256])
        
        combined_attn_logits = torch.stack(attn_logits_list, dim=0)  # torch.Size([4, 64, 1, 196, 6]) sbilk
        fused_attn_logits = torch.einsum('sbik,sbilk->bilk', scale_weights_masks, combined_attn_logits)  # torch.Size([64, 1, 196, 6])

        return fused_slots, fused_attention_masks, fused_init_slots, fused_attn_logits  

  
class TransformerConnector(nn.Module):
    """
        Module for fusing slots based on a self-attention mechanism. Here, each slot of each layer is
        treated as an independent sequence; and a self-attention mechanism produces one final slot
        tensor based on these sequences. The attention maps, logits and init_slots are fused through
        one of the above fusers.
    """

    def __init__(self, slot_dim, num_layers, num_heads=4, ff_dim=512, dropout=0.1):
        """
            slot_dim: Dimension of each slot.
            num_heads: Number of attention heads.
            ff_dim: Dimension of the feed-forward layer.
            dropout: Dropout rate for regularization.
        """
        super(TransformerConnector, self).__init__()
        
        # Positional encoding for distinguishing slots across layers as learnable parameter
        self.positional_encoding = nn.Parameter(torch.randn(1, 1, slot_dim))
        
        # The transformer module itself
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=slot_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )

        # For attention map, init slot and attn logits fusion
        self.map_fuser = MapFusionThroughContribution(196, num_layers)
    
    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):
        """
            slot_list: List of slots [B, num_slots, slot_dim], e.g. VOC [64, 6, 256] with len = num_layers
            slot_att_list: List of slot attention maps [B, H_emb * W_emb, num_slots], e.g. VOC [64, 196, 6] with len = num_layers
            init_slot_list: List of slot initializations [B, num_slots, slot_dim] with len = num_layers
            attn_logits_list: List of attention logits [B, 1, H_emb * W_emb, num_slots]) with len = num_layers
        """
        # Stack slots along the layer dimension: [B, num_layers, num_slots, slot_dim]
        stacked_slots = torch.stack(slot_list, dim=1)
        
        # Reshape for Transformer, make it so that every slot of every scale is treated as independent sequence: [B, num_layers * num_slots, slot_dim]
        batch_size, num_layers, num_slots, slot_dim = stacked_slots.shape
        reshaped_slots = stacked_slots.view(batch_size, num_layers * num_slots, slot_dim)
        
        # Add positional encoding to distinguish slot positions
        slots_with_position = reshaped_slots + self.positional_encoding  # [B, num_layers * num_slots, slot_dim]
        
        # Pass through Transformer
        transformed_slots = self.transformer_encoder(slots_with_position)  # [B, num_layers * num_slots, slot_dim]
        
        # Reshape back to original slot shape [B, num_slots, slot_dim]
        fused_slots = transformed_slots.view(batch_size, num_layers, num_slots, slot_dim).mean(dim=1)
        
        # Fuse attention maps, init_slots and logits accordingly
        fused_attn, fused_inits, fused_logits = self.map_fuser(fused_slots, slot_list, slot_att_list, init_slot_list, attn_logits_list)
        
        return fused_slots, fused_attn, fused_inits, fused_logits 
    

class DenseConnector(nn.Module):
    """
        Implements fusion of slots from multiple layers of the vision encoder through a learned projeciton
        similar to DenseConnector, as described here:

        https://openreview.net/pdf?id=Ioabr42B44

        Here, slots are concatenated either along the patch (1) or feature (2+3) dimension and fed through a projection
        layer to obtain a fused slot representation with the same shape as a slot tensor of an individual SA module. The
        dense channel integration concats pair-wise sums of slots prior to projection. For now, only channel integration
        was implemented here.

        To align the slot attention maps (and therefore init_slots and logits), we calculate the individual contribution
        of every slot of every layer to the final fused slot through cosine similarity, and weigh the attention maps by
        its individual components per slot by this weight.

        EXAMPLE: For three layers of SA with slots x1, x2, x3[B, 6, 256] and attention maps [B, 196, 6]:

        (1) Sparse token integration - Concat(x1, x2, x3, dim=-2) [B, 18, 256], Projection -> [B, 6, 256]
        (2) Sparse channel integration - Concat(x1, x2, x3, dim=-1) [B, 6, 256 * 3], Projection -> [B, 6, 256]
        (3) Dense channel integration - Concat((x1 + x2), (x2 + x3), dim=-1) [B, 6, 256 * 2], Projection -> [B, 6, 256]
    """

    def __init__(self, slot_dim, num_layers, dc_type="dense", mlp_depth=1):
        super().__init__()

        # Properly parse type
        if dc_type == "dense":
            self.dense = True
        elif dc_type == "sparse":
            self.dense = False
        else:
            raise ValueError(f"The DenseConnector integration type has to be 'sparse' or 'dense', but got {dc_type}")
        
        # Store parameters
        self.num_layers = num_layers - (1 if self.dense else 0)
        self.hidden_size = slot_dim * 3     # In accordance to hidden_size * 3
        self.mlp_depth = mlp_depth

        # Init MLP for channel integration
        modules = [nn.Linear(slot_dim * self.num_layers, self.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.hidden_size, self.hidden_size))
        modules.append(nn.GELU())
        modules.append(nn.Linear(self.hidden_size, slot_dim))
        # modules.append(nn.LayerNorm(slot_dim))
        self.mlp = nn.Sequential(*modules)

        # For attention map, init slot and attn logits fusion
        self.map_fuser = MapFusionWithLearnedWeights(196, self.num_layers)
        # self.map_fuser = MapFusionWithLearnedWeights(196, self.num_layers + self.num_layers - 1)

    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):
        """
            slot_list: List of slots [B, num_slots, slot_dim], e.g. VOC [64, 6, 256] with len = num_layers
            slot_att_list: List of slot attention maps [B, H_emb * W_emb, num_slots], e.g. VOC [64, 196, 6] with len = num_layers
            init_slot_list: List of slot initializations [B, num_slots, slot_dim] with len = num_layers
            attn_logits_list: List of attention logits [B, 1, H_emb * W_emb, num_slots]) with len = num_layers
        """
        
        # Add pair-wise [x_i + x_(i+1)] to concat list if we have dense integration, for all lists
        if self.dense:
            slot_list = [(slot_list[i] + slot_list[i + 1]) / 2 for i in range(self.num_layers)]
            slot_att_list = [(slot_att_list[i] + slot_att_list[i + 1]) / 2 for i in range(self.num_layers)]
            init_slot_list = [(init_slot_list[i] + init_slot_list[i + 1]) / 2 for i in range(self.num_layers)]
            attn_logits_list = [(attn_logits_list[i] + attn_logits_list[i + 1]) / 2 for i in range(self.num_layers)]

        # Concat along feature dimension to [B, num_slots, slot_dim * num_layers (num_layers - 1 if dense)]
        concat = torch.concat(slot_list, dim=-1)

        # Project concat down to original shape [B, num_slots, slot_dim]
        fused_slots = self.mlp(concat)

        # Fuse attention maps, init_slots and logits accordingly
        fused_attn, fused_inits, fused_logits = self.map_fuser(fused_slots, slot_list, slot_att_list, init_slot_list, attn_logits_list)
        return fused_slots, fused_attn, fused_inits, fused_logits 
    
