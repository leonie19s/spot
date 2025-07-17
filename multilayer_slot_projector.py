import torch
import torch.nn as nn
import torch.nn.functional as F
import gc


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
    def __init__(self, slot_dim, num_layers, num_slots, fct=None):
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

    def __init__(self, slot_dim, num_layers, num_slots, num_heads=4, ff_dim=512, dropout=0.1):
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

    def __init__(self, slot_dim, num_layers, num_slots, dc_type="dense", mlp_depth=1):
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
    

class GatedFusion(nn.Module):

    def __init__(self, slot_dim, num_layers, num_slots):
        super().__init__()
        self.L = num_layers
        
        # Use one small MLP per layer that predicts for a given input a gate weight for each slot for every layer
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 32, bias=True), # nn.Linear(slot_dim, slot_dim//2, bias=True),
                nn.LayerNorm(32), # nn.LayerNorm(slot_dim//2),
                nn.GELU(),
                nn.Linear(32, 1, bias=True) #nn.Linear(slot_dim//2, 1, bias=True)
            )
            for _ in range(self.L)
        ])

        # Gate softmax temperature: T < 1 makes the distribution sharper (more “hard” selection single layer) and T > 1 makes it flatter.
        self.softmax_temp = 2
        
        # Projection heads for concatenated outputs
        self.slot_proj = nn.Linear(self.L * slot_dim, slot_dim, bias=True)

        # Record epoch-wise per-layer gated weights
        self.per_layer_weight = [[] for _ in range(self.L)]

    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):

        # Compute a scalar gate weight for each slot over the layers, i.e. that weighs each slot s_0 .. s_slots across layers 0, ..., L
        # Output: List of L tensors [B, S, 1]
        gate_logits = []
        for l in range(self.L):
            # Gating with mean-pooling over slot channel dim
            pool = slot_list[l].mean(dim=-1, keepdim=True)  # Mean-pool over the channel dimension to feed into MLP: [B, S, C] -> [B, S, 1]
            logit = self.gate_mlps[l](pool)     
            
            # Gating without mean-pooling over slot channel dim
            # logit = self.gate_mlps[l](slot_list[l])    
            
            gate_logits.append(logit)

        # Stack and softmax over the layer dimension so that gated weights sum to 1 -> [B, S, L]
        # G basically gives the importance of layer l for slot s in sample b
        G = torch.stack(gate_logits, dim=2).squeeze(-1).to(slot_list[0].device)
        G = F.softmax(G / self.softmax_temp, dim=2)

        # One could also introduce a temperature here for: T < 1 making the distribution sharper (more “hard” selection
        # of a single layer) and T > 1 making it flatter.
        # G = F.softmax(G / temperature, dim=2)

        # Apply gates to the outputs of each layer
        gated_vecs  = []
        gated_masks = []
        gated_masks_logits = []
        for l in range(self.L):
            g = G[:, :, l].unsqueeze(-1)     # [B, S, 1]
            self.per_layer_weight[l].append(g.detach().cpu())
            # fuse slot vectors
            gated_vecs.append(g * slot_list[l])    # [B, S, C]
            # fuse slot masks (broadcast over P)
            gated_masks.append(g.permute(0,2,1) * slot_att_list[l])
            # fuse slot logit masks
            gated_masks_logits.append(g.permute(0, 2, 1).unsqueeze(1) * attn_logits_list[l])


        # Concatenate the slots across layers in channel dimension and project down to original size
        V_cat = torch.cat(gated_vecs, dim=-1)           # [B, S, L*C]
        S_fused = self.slot_proj(V_cat)                 # [B, S, C]

        # Linear fusion of gated masks by simply summing them together instead of projection head to maintain semantics of attention map
        A_fused = sum(gated_masks)
        A_logits = sum(gated_masks_logits)

        # Fusion of logits, use them to create attention maps again
        # A_logits = torch.stack(gated_masks_logits, dim=0).sum(dim=0)  # [B,1,P,S]
        # A_fused = F.softmax(A_logits, dim=-1).squeeze()  # [B,1,P,S] or squeeze to [B,P,S], Softmax over slots (last dim) to get a valid attention map

        # TODO: Do init slot list as well, even though it is not strictly necessary
        # TODO: Maybe do channel-wise gating as well? So far, every channel is weighted equally important

        return S_fused, A_fused, init_slot_list[0], A_logits
    
    def get_mean_gates(self, std=False):
        means = []
        stds = []
        for l in range(self.L):
            per_batch_list = self.per_layer_weight[l]

            # Concat along all data samples, and then collapse there
            all_g = torch.cat(per_batch_list, dim=0)
            means.append(all_g.mean().item())
            stds.append(all_g.std().item())
        
        # Reset for this epoch to avoid RAM issues
        self.per_layer_weight = [[] for _ in range(self.L)]
        gc.collect()

        if std:
            return means,stds
        return means


class ChannelwiseGatedFusion(nn.Module):

    def __init__(self, slot_dim, num_layers, num_slots, hidden_dim=32):
        super().__init__()
        self.L = num_layers
        
        # one MLP per layer to predict C channel-wise logits per slot
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(slot_dim, hidden_dim, bias=True),
                nn.GELU(),
                nn.Linear(hidden_dim, slot_dim, bias=True)
            )
            for _ in range(self.L)
        ])
        
        self.slot_proj = nn.Linear(self.L * slot_dim, slot_dim, bias=True)
        self.mask_proj = self.mask_proj = nn.Linear(self.L * num_slots, num_slots, bias=True)

    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):

        # 1) compute channel-wise logits per layer → list of [B, S, C]
        gate_logits = []
        for l in range(self.L):
            # slot_list[l]: [B, S, C]
            logit = self.gate_mlps[l](slot_list[l])  # [B, S, C]
            gate_logits.append(logit)

        # 2) stack into [B, S, L, C] and softmax over L
        G = torch.stack(gate_logits, dim=2)         # [B, S, L, C]
        G = F.softmax(G, dim=2)                     # sum over layers = 1 for each (b,s,c)

        # 3) apply gates channel-wise
        gated_vecs  = []
        gated_masks = []
        gated_logits= []
        for l in range(self.L):
            g_l = G[:, :, l, :]                     # [B, S, C]
            # fuse slot vectors (channel-wise)
            gated_vecs.append(g_l * slot_list[l])   # [B, S, C]
            # fuse post‐softmax masks (reduce g_l to scalar per slot)
            s = g_l.mean(dim=-1, keepdim=True)      # [B, S, 1]
            gated_masks.append(s.permute(0,2,1) * slot_att_list[l])
            # fuse pre-softmax logits           # [B,1,S,1]
            gated_logits.append(s.permute(0, 2, 1).unsqueeze(1) * attn_logits_list[l])

        # 4a) concat & project slot vectors
        V_cat = torch.cat(gated_vecs, dim=-1)       # [B, S, L*C]
        S_fused = self.slot_proj(V_cat)             # [B, S, C]

        # 4b) concat & project masks
        A_cat = torch.cat(gated_masks, dim=-1)      # [B, P, L*S]
        A_logits = self.mask_proj(A_cat)            # [B, P, S]
        A_fused  = F.softmax(A_logits, dim=-1)

        # 4c) sum gated_logits
        fused_logits = torch.stack(gated_logits, dim=0).sum(dim=0)  # [B,1,P,S]

        return S_fused, A_fused, init_slot_list[0], fused_logits

    def get_mean_gates(self):
        return [1, 1, 1, 1]
    

class GatedFusionSingleMLP(nn.Module):

    def __init__(self, slot_dim, num_layers, num_slots):
        super().__init__()
        self.L = num_layers
        
        # Use one small MLP per layer that predicts for a given input a gate weight for each slot for every layer
        self.gate_mlp = nn.Sequential(
                nn.Linear(self.L, 64, bias=True), # nn.Linear(slot_dim, slot_dim//2, bias=True),
                nn.LayerNorm(64), # nn.LayerNorm(slot_dim//2),
                nn.GELU(),
                nn.Linear(64, self.L, bias=True) #nn.Linear(slot_dim//2, 1, bias=True)
            )

        # Gate softmax temperature: T < 1 makes the distribution sharper (more “hard” selection single layer) and T > 1 makes it flatter.
        self.softmax_temp = 1
        
        # Projection heads for concatenated outputs
        self.slot_proj = nn.Linear(self.L * slot_dim, slot_dim, bias=True)

        # Record epoch-wise per-layer gated weights
        self.per_layer_weight = [[] for _ in range(self.L)]

    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):

        # Compute a scalar gate weight for each slot over the layers, i.e. that weighs each slot s_0 .. s_slots across layers 0, ..., L
        # Output: List of L tensors [B, S, 1]
        slot_list_mean_pooled = [slot_list[l].mean(dim=-1, keepdim=True) for l in range(self.L)] # List of [B, S, 1]
        gate_logits = self.gate_mlp(torch.concat(slot_list_mean_pooled, dim=-1)).to(slot_list[0].device) # Stack to [B, S, L], obtain logits of [B, S, L]
        G = F.softmax(gate_logits / self.softmax_temp, dim=2)

        # Apply gates to the outputs of each layer
        gated_vecs  = []
        gated_masks = []
        gated_masks_logits = []
        for l in range(self.L):
            g = G[:, :, l].unsqueeze(-1)     # [B, S, 1]
            self.per_layer_weight[l].append(g.detach().cpu())
            # fuse slot vectors
            gated_vecs.append(g * slot_list[l])    # [B, S, C]
            # fuse slot masks (broadcast over P)
            gated_masks.append(g.permute(0,2,1) * slot_att_list[l])
            # fuse slot logit masks
            gated_masks_logits.append(g.permute(0, 2, 1).unsqueeze(1) * attn_logits_list[l])


        # Concatenate the slots across layers in channel dimension and project down to original size
        V_cat = torch.cat(gated_vecs, dim=-1)           # [B, S, L*C]
        S_fused = self.slot_proj(V_cat)                 # [B, S, C]

        # Linear fusion of gated masks by simply summing them together instead of projection head to maintain semantics of attention map
        A_fused = sum(gated_masks)
        A_logits = sum(gated_masks_logits)

        # Fusion of logits, use them to create attention maps again
        # A_logits = torch.stack(gated_masks_logits, dim=0).sum(dim=0)  # [B,1,P,S]
        # A_fused = F.softmax(A_logits, dim=-1).squeeze()  # [B,1,P,S] or squeeze to [B,P,S], Softmax over slots (last dim) to get a valid attention map

        # TODO: Do init slot list as well, even though it is not strictly necessary
        # TODO: Maybe do channel-wise gating as well? So far, every channel is weighted equally important

        return S_fused, A_fused, init_slot_list[0], A_logits
    
    def get_mean_gates(self, std=False):
        means = []
        stds = []
        for l in range(self.L):
            per_batch_list = self.per_layer_weight[l]

            # Concat along all data samples, and then collapse there
            all_g = torch.cat(per_batch_list, dim=0)
            means.append(all_g.mean().item())
            stds.append(all_g.std().item())
        
        # Reset for this epoch to avoid RAM issues
        self.per_layer_weight = [[] for _ in range(self.L)]
        gc.collect()

        if std:
            return means,stds
        return means
    

class GatedFusionNoPooling(nn.Module):

    def __init__(self, slot_dim, num_layers, num_slots):
        super().__init__()
        self.L = num_layers
        
        # Use one small MLP per layer that predicts for a given input a gate weight for each slot for every layer
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(slot_dim),
                nn.Linear(slot_dim, 128),
                nn.GELU(),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )
            for _ in range(self.L)
        ])

        # Gate softmax temperature: T < 1 makes the distribution sharper (more “hard” selection single layer) and T > 1 makes it flatter.
        self.softmax_temp = 1
        
        # Projection heads for concatenated outputs
        self.slot_proj = nn.Linear(self.L * slot_dim, slot_dim, bias=True)

        # Record epoch-wise per-layer gated weights
        self.per_layer_weight = [[] for _ in range(self.L)]

    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):

        # Compute a scalar gate weight for each slot over the layers, i.e. that weighs each slot s_0 .. s_slots across layers 0, ..., L
        # Output: List of L tensors [B, S, 1]
        gate_logits = [self.gate_mlps[l](slot_list[l]) for l in range(self.L)] # No mean pooling

        # Stack and softmax over the layer dimension so that gated weights sum to 1 -> [B, S, L]
        # G basically gives the importance of layer l for slot s in sample b
        G = torch.stack(gate_logits, dim=2).squeeze(-1).to(slot_list[0].device)
        G = F.softmax(G / self.softmax_temp, dim=2)

        # One could also introduce a temperature here for: T < 1 making the distribution sharper (more “hard” selection
        # of a single layer) and T > 1 making it flatter.
        # G = F.softmax(G / temperature, dim=2)

        # Apply gates to the outputs of each layer
        gated_vecs  = []
        gated_masks = []
        gated_masks_logits = []
        for l in range(self.L):
            g = G[:, :, l].unsqueeze(-1)     # [B, S, 1]
            self.per_layer_weight[l].append(g.detach().cpu())
            # fuse slot vectors
            gated_vecs.append(g * slot_list[l])    # [B, S, C]
            # fuse slot masks (broadcast over P)
            gated_masks.append(g.permute(0,2,1) * slot_att_list[l])
            # fuse slot logit masks
            gated_masks_logits.append(g.permute(0, 2, 1).unsqueeze(1) * attn_logits_list[l])


        # Concatenate the slots across layers in channel dimension and project down to original size
        V_cat = torch.cat(gated_vecs, dim=-1)           # [B, S, L*C]
        S_fused = self.slot_proj(V_cat)                 # [B, S, C]

        # Linear fusion of gated masks by simply summing them together instead of projection head to maintain semantics of attention map
        A_fused = sum(gated_masks)
        A_logits = sum(gated_masks_logits)

        # Fusion of logits, use them to create attention maps again
        # A_logits = torch.stack(gated_masks_logits, dim=0).sum(dim=0)  # [B,1,P,S]
        # A_fused = F.softmax(A_logits, dim=-1).squeeze()  # [B,1,P,S] or squeeze to [B,P,S], Softmax over slots (last dim) to get a valid attention map

        # TODO: Do init slot list as well, even though it is not strictly necessary
        # TODO: Maybe do channel-wise gating as well? So far, every channel is weighted equally important

        return S_fused, A_fused, init_slot_list[0], A_logits
    
    def get_mean_gates(self, std=False):
        means = []
        stds = []
        for l in range(self.L):
            per_batch_list = self.per_layer_weight[l]

            # Concat along all data samples, and then collapse there
            all_g = torch.cat(per_batch_list, dim=0)
            means.append(all_g.mean().item())
            stds.append(all_g.std().item())
        
        # Reset for this epoch to avoid RAM issues
        self.per_layer_weight = [[] for _ in range(self.L)]
        gc.collect()

        if std:
            return means,stds
        return means


class GatedFusionNoSoftmax(nn.Module):

    def __init__(self, slot_dim, num_layers, num_slots):
        super().__init__()
        self.L = num_layers
        
        # Use one small MLP per layer that predicts for a given input a gate weight for each slot for every layer
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, 32, bias=True), # nn.Linear(slot_dim, slot_dim//2, bias=True),
                nn.LayerNorm(32), # nn.LayerNorm(slot_dim//2),
                nn.GELU(),
                nn.Linear(32, 1, bias=True) #nn.Linear(slot_dim//2, 1, bias=True)
            )
            for _ in range(self.L)
        ])
        
        # Projection heads for concatenated outputs
        self.slot_proj = nn.Linear(self.L * slot_dim, slot_dim, bias=True)

        # Record epoch-wise per-layer gated weights
        self.per_layer_weight = [[] for _ in range(self.L)]

    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):

        # Compute a scalar gate weight for each slot over the layers, i.e. that weighs each slot s_0 .. s_slots across layers 0, ..., L
        # Output: List of L tensors [B, S, 1]
        gate_logits = []
        for l in range(self.L):
            # Gating with mean-pooling over slot channel dim
            pool = slot_list[l].mean(dim=-1, keepdim=True)  # Mean-pool over the channel dimension to feed into MLP: [B, S, C] -> [B, S, 1]
            logit = self.gate_mlps[l](pool)     
            
            # Gating without mean-pooling over slot channel dim
            # logit = self.gate_mlps[l](slot_list[l])    
            
            gate_logits.append(logit)

        G = torch.stack(gate_logits, dim=2).squeeze(-1).to(slot_list[0].device)

        # Apply gates to the outputs of each layer
        gated_vecs  = []
        gated_masks = []
        gated_masks_logits = []
        for l in range(self.L):
            g = G[:, :, l].unsqueeze(-1)     # [B, S, 1]
            self.per_layer_weight[l].append(g.detach().cpu())
            # fuse slot vectors
            gated_vecs.append(g * slot_list[l])    # [B, S, C]
            # fuse slot masks (broadcast over P)
            gated_masks.append(g.permute(0,2,1) * slot_att_list[l])
            # fuse slot logit masks
            gated_masks_logits.append(g.permute(0, 2, 1).unsqueeze(1) * attn_logits_list[l])


        # Concatenate the slots across layers in channel dimension and project down to original size
        V_cat = torch.cat(gated_vecs, dim=-1)           # [B, S, L*C]
        S_fused = self.slot_proj(V_cat)                 # [B, S, C]

        # Linear fusion of gated masks by simply summing them together instead of projection head to maintain semantics of attention map
        A_fused = sum(gated_masks)
        A_logits = sum(gated_masks_logits)
        A_fused = F.softmax(A_fused, dim=1)  # Softmax attention map over patches to ensure its valid
        A_logits = F.softmax(A_fused, dim=1)  # Softmax attention map over patches to ensure its valid

        # Fusion of logits, use them to create attention maps again
        # A_logits = torch.stack(gated_masks_logits, dim=0).sum(dim=0)  # [B,1,P,S]
        # A_fused = F.softmax(A_logits, dim=-1).squeeze()  # [B,1,P,S] or squeeze to [B,P,S], Softmax over slots (last dim) to get a valid attention map

        # TODO: Do init slot list as well, even though it is not strictly necessary
        # TODO: Maybe do channel-wise gating as well? So far, every channel is weighted equally important

        return S_fused, A_fused, init_slot_list[0], A_logits
    
    def get_mean_gates(self, std=False):
        means = []
        stds = []
        for l in range(self.L):
            per_batch_list = self.per_layer_weight[l]

            # Concat along all data samples, and then collapse there
            all_g = torch.cat(per_batch_list, dim=0)
            means.append(all_g.mean().item())
            stds.append(all_g.std().item())
        
        # Reset for this epoch to avoid RAM issues
        self.per_layer_weight = [[] for _ in range(self.L)]
        gc.collect()

        if std:
            return means,stds
        return means
    

class GatedFusionLayerWise(nn.Module):

    def __init__(self, slot_dim, num_layers, num_slots):
        super().__init__()
        self.L = num_layers
        
        # Use one small MLP per layer that predicts for a given input a gate weight for every layer, disregarding the slots
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(slot_dim, slot_dim//2, bias=True),
                nn.LayerNorm(slot_dim//2),
                nn.GELU(),
                nn.Linear(slot_dim//2, 1, bias=True)
            )
            for _ in range(self.L)
        ])

        # Gate softmax temperature: T < 1 makes the distribution sharper (more “hard” selection single layer) and T > 1 makes it flatter.
        self.softmax_temp = 2
        
        # Projection heads for concatenated outputs
        self.slot_proj = nn.Linear(self.L * slot_dim, slot_dim, bias=True)

        # Record epoch-wise per-layer gated weights
        self.per_layer_weight = [[] for _ in range(self.L)]

    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):

        # Compute a scalar gate weight for each layer, Output: List of L tensors [B, 1]
        gate_logits = []
        for l in range(self.L):
            pool = slot_list[l].mean(dim=1, keepdim=True)  # Mean-pool over the slot dimension to feed into MLP: [B C] -> [B, S, 1]
            logit = self.gate_mlps[l](pool)     
            gate_logits.append(logit)

        # Stack and softmax over the layer dimension so that gated weights sum to 1 -> [B, L]
        # G basically gives the importance of layer l in sample b
        G = torch.stack(gate_logits, dim=1).squeeze(-1).to(slot_list[0].device)
        G = F.softmax(G / self.softmax_temp, dim=1)

        # Apply gates to the outputs of each layer
        gated_vecs  = []
        gated_masks = []
        gated_masks_logits = []
        for l in range(self.L):
            g = G[:, l].unsqueeze(-1)     # [B, 1]
            self.per_layer_weight[l].append(g.detach().cpu())
            # fuse slot vectors
            gated_vecs.append(g * slot_list[l])    # [B, S, C]
            # fuse slot masks (broadcast over P)
            gated_masks.append(g.permute(0,2,1) * slot_att_list[l])
            # fuse slot logit masks
            gated_masks_logits.append(g.permute(0, 2, 1).unsqueeze(1) * attn_logits_list[l])


        # Concatenate the slots across layers in channel dimension and project down to original size
        V_cat = torch.cat(gated_vecs, dim=-1)           # [B, S, L*C]
        S_fused = self.slot_proj(V_cat)                 # [B, S, C]

        # Linear fusion of gated masks by simply summing them together instead of projection head to maintain semantics of attention map
        A_fused = sum(gated_masks)
        A_logits = sum(gated_masks_logits)

        return S_fused, A_fused, init_slot_list[0], A_logits
    
    def get_mean_gates(self, std=False):
        means = []
        stds = []
        for l in range(self.L):
            per_batch_list = self.per_layer_weight[l]

            # Concat along all data samples, and then collapse there
            all_g = torch.cat(per_batch_list, dim=0)
            means.append(all_g.mean().item())
            stds.append(all_g.std().item())
        
        # Reset for this epoch to avoid RAM issues
        self.per_layer_weight = [[] for _ in range(self.L)]
        gc.collect()

        if std:
            return means,stds
        return means