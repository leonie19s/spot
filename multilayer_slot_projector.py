import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial



class MapFusionWithLearnedWeights(nn.Module):
    """
        Module for fusing the attention maps, init slots and logits together through a linear combination
        with learned weights that sum to 1.
    """
    
    def __init__(self, num_layers):
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

    def __init__(self, num_layers):
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


class TransformerConnector(nn.Module):
    def __init__(self, slot_dim, num_heads, ff_dim, dropout=0.1):
        """
            Initialize the Transformer-based SlotFusion module.
            slot_dim: Dimension of each slot (slot_dim).
            num_heads: Number of attention heads in the Transformer.
            ff_dim: Dimension of the feed-forward layer in the Transformer.
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
    
    def forward(self, slot_list):
        """
            TODO
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
        
        return fused_slots
    

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

    def __init__(self, slot_dim, num_layers, dc_type, mlp_depth):
        super().__init__()

        # Properly parse type
        if dc_type == "dense":
            self.dense = True
        elif dc_type == "sparse":
            self.dense = False
        else:
            raise ValueError(f"The DenseConnector integration type has to be 'sparse' or 'dense', but got {dc_type}")
        
        # Store parameters
        self.num_layers = num_layers
        self.hidden_size = slot_dim * 3     # In accordance to hidden_size * 3
        self.mlp_depth = mlp_depth

        # Init MLP for channel integration
        modules = [nn.Linear(slot_dim * (num_layers - (1 if self.dense else 0)), self.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.hidden_size, self.hidden_size))
        modules.append(nn.GELU())
        modules.append(nn.Linear(self.hidden_size, slot_dim))
        # modules.append(nn.LayerNorm(slot_dim))
        self.mlp = nn.Sequential(*modules)

        # For attention map, init slot and attn logits fusion
        self.map_fuser = MapFusionWithLearnedWeights(num_layers)

    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):
        """
            slot_list: List of slots [B, num_slots, slot_dim], e.g. VOC [64, 6, 256] with len = num_layers
            slot_att_list: List of slot attention maps [B, H_emb * W_emb, num_slots], e.g. VOC [64, 196, 6] with len = num_layers
            init_slot_list: List of slot initializations [B, num_slots, slot_dim] with len = num_layers
            attn_logits_list: List of attention logits [B, 1, H_emb * W_emb, num_slots]) with len = num_layers
        """
        
        # Add pair-wise [x_i + x_(i+1)] to concat list if we have dense integration, for all lists
        if self.dense:
            slot_list = [(slot_list[i] + slot_list[i + 1]) / 2 for i in range(self.num_layers - 1)]
            slot_att_list = [(slot_att_list[i] + slot_att_list[i + 1]) / 2 for i in range(self.num_layers - 1)]
            init_slot_list = [(init_slot_list[i] + init_slot_list[i + 1]) / 2 for i in range(self.num_layers - 1)]
            attn_logits_list = [(attn_logits_list[i] + attn_logits_list[i + 1]) / 2 for i in range(self.num_layers - 1)]

        # Concat along feature dimension to [B, num_slots, slot_dim * num_layers (num_layers - 1 if dense)]
        concat = torch.concat(slot_list, dim=-1)

        # Project concat down to original shape [B, num_slots, slot_dim]
        fused_slots = self.mlp(concat)

        # Fuse attention maps, init_slots and logits accordingly
        fused_attn, fused_inits, fused_logits = self.map_fuser(fused_slots, slot_list, slot_att_list, init_slot_list, attn_logits_list)
        return fused_slots, fused_attn, fused_inits, fused_logits 
    

class MSCrossAttnBlock(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=16, n_points=4, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 dropout=0.1, init_values=0.):
        
        from deformable_attention.ops.modules import MSDeformAttn

        super().__init__()
        self.select_layer = [_ for _ in range(n_levels)]
        self.query_layer = -1 

        self.cross_attn = MSDeformAttn(d_model=d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.query_norm = norm_layer(d_model)
        self.feat_norm = norm_layer(d_model)
        self.gamma1 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)

        self.norm1 = norm_layer(d_model)
        self.self_attn = MSDeformAttn(d_model=d_model, n_levels=1, n_heads=n_heads, n_points=n_points)
        self.gamma2 = nn.Parameter(init_values * torch.ones((d_model)), requires_grad=True)

        self._reset_parameters()

    def _reset_parameters(self):
        from deformable_attention.ops.modules import MSDeformAttn
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]
        return reference_points

    def forward(self, srcs, masks=None, pos_embeds=None):
        # prepare input feat
        src_flatten = []
        spatial_shapes = []
        for lvl in self.select_layer: 
            src = srcs[lvl] # .permute(0, 2, 1)
            _, hw, _ = src.shape    # Shape for slots: [64, 6, 256]
            e = int(math.sqrt(hw))
            spatial_shape = (e, e)
            spatial_shapes.append(spatial_shape)
            src_flatten.append(src)
        feat = torch.cat(src_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # cross attn
        pos = None  # TODO
        query = srcs[self.query_layer] # .permute(0, 2, 1)
        query = self.with_pos_embed(query, pos)  # bs, h*w, c
        query_e = int(math.sqrt(query.shape[1]))  # h == w

        reference_points = self.get_reference_points([(query_e, query_e)], device=query.device)
        attn = self.cross_attn(self.query_norm(query), reference_points, self.feat_norm(feat), spatial_shapes,
                               level_start_index, None)

        # self attn
        attn1 = self.norm1(attn)
        attn_pos = None  # TODO
        spatial_shapes_attn = torch.as_tensor([(query_e, query_e)], dtype=torch.long, device=attn1.device)
        level_start_index_attn = torch.cat(
            (spatial_shapes_attn.new_zeros((1,)), spatial_shapes_attn.prod(1).cumsum(0)[:-1]))
        reference_points_attn = self.get_reference_points(spatial_shapes_attn, device=attn1.device)
        attn2 = self.self_attn(self.with_pos_embed(attn1, attn_pos), reference_points_attn, attn1, spatial_shapes_attn,
                               level_start_index_attn, None)
        attn = attn + self.gamma2 * attn2

        # Residual Connection
        tgt = query + self.gamma1 * attn

        return tgt
    

class MMFuser(nn.Module):
    """
        TODO
    """

    def __init__(self, slot_dim, num_layers, dc_type, mlp_depth):
        super().__init__()

        self.hidden_size = slot_dim * 3
        modules = [
            MSCrossAttnBlock(n_levels=num_layers, d_model=slot_dim),
            nn.Linear(slot_dim, self.hidden_size)
        ]

        # Init MLP for channel integration
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(self.hidden_size, self.hidden_size))
        modules.append(nn.GELU())
        modules.append(nn.Linear(self.hidden_size, slot_dim))
        # modules.append(nn.LayerNorm(slot_dim))
        self.mlp = nn.Sequential(*modules)

    def forward(self, slot_list, slot_att_list, init_slot_list, attn_logits_list):
        """
            slot_list: List of slots [B, num_slots, slot_dim], e.g. VOC [64, 6, 256] with len = num_layers
            slot_att_list: List of slot attention maps [B, H_emb * W_emb, num_slots], e.g. VOC [64, 196, 6] with len = num_layers
            init_slot_list: List of slot initializations [B, num_slots, slot_dim] with len = num_layers
            attn_logits_list: List of attention logits [B, 1, H_emb * W_emb, num_slots]) with len = num_layers
        """

        # Concat along feature dimension to [B, num_slots, slot_dim * num_layers (num_layers - 1 if dense)]
        # concat = torch.concat(slot_list, dim=-1)

        # Project concat down to original shape [B, num_slots, slot_dim]
        fused_slots = self.mlp(slot_list)

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

        return fused_slots, fused_attn, fused_inits, fused_logits 

