''' Based on SLATE and BOQSA libraries:
https://github.com/singhgautam/slate/blob/master/slot_attn.py
https://github.com/YuLiu-LY/BO-QSA/blob/main/models/slot_attn.py
'''

from utils_spot import *
from timm.models.layers import DropPath
from ocl_metrics import unsupervised_mask_iou
import matplotlib.pyplot as plt
from matplotlib import cm
from utils_spot import visualize_layer_attn
from multilayer_slot_projector import DenseConnector, SimpleConnector, TransformerConnector, NormWeightConnector
from functools import partial


# Constants
DO_PLOT_ORDERING = False
LAYER_ATTN_VIS = False
FUSION_STRING_MAPPING = {
    "mean": partial(SimpleConnector, fct=torch.mean),
    "sum": partial(SimpleConnector, fct=torch.sum),
    "max": partial(SimpleConnector, fct=lambda x, dim: torch.max(x, dim=dim).values),
    "residual": partial(SimpleConnector, fct=lambda x, dim: x[-1]),
    "norm_weight": NormWeightConnector,
    "denseconnector": DenseConnector,
    "transformerconnector": TransformerConnector
}


class SlotAttention(nn.Module):
    def __init__(
        self,
        num_iter,
        input_size,
        slot_size, 
        mlp_size, 
        truncate,
        heads,
        epsilon=1e-8, 
        drop_path=0,
    ):
        super().__init__()
        self.num_iter = num_iter
        self.input_size = input_size
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.truncate = truncate
        self.num_heads = heads

        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)
        
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.mlp = nn.Sequential(
            linear(slot_size, mlp_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_size, slot_size),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        print(self.truncate)
        assert self.truncate in ['bi-level', 'fixed-point', 'none']


    def forward(self, inputs, slots_init):
        # `inputs` has shape [batch_size, num_inputs, input_size].
        # `slots` has shape [batch_size, num_slots, slot_size].
        slots = slots_init
        B, N_kv, D_inp = inputs.size()
        B, N_q, D_slot = slots.size()

        inputs = self.norm_inputs(inputs)
        #print all types
        #print("inputs",type(inputs), "slots init", type(slots_init),"num heads", type(self.num_heads), "B",type(B), "N_kv", type(N_kv),"Nq", type(N_q), "D_inp",type(D_inp),"D_slot", type(D_slot))
        k = self.project_k(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        v = self.project_v(inputs).view(B, N_kv, self.num_heads, -1).transpose(1, 2)    # Shape: [batch_size, num_heads, num_inputs, slot_size // num_heads].
        k = ((self.slot_size // self.num_heads) ** (-0.5)) * k
        
        # Multiple rounds of attention.
        for i in range(self.num_iter):
            if i == self.num_iter  - 1:
                if self.truncate == 'bi-level':
                    slots = slots.detach() + slots_init - slots_init.detach()
                elif self.truncate == 'fixed-point':
                    slots = slots.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)
            
            # Attention.
            q = self.project_q(slots).view(B, N_q, self.num_heads, -1).transpose(1, 2)  # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            attn_logits = torch.matmul(k, q.transpose(-1, -2))                          # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn = F.softmax(
                attn_logits.transpose(1, 2).reshape(B, N_kv, self.num_heads * N_q)
            , dim=-1).view(B, N_kv, self.num_heads, N_q).transpose(1, 2)                # Shape: [batch_size, num_heads, num_inputs, num_slots].
            attn_vis = attn.sum(1)                                                      # Shape: [batch_size, num_inputs, num_slots].
            
            # Weighted mean.
            attn = attn + self.epsilon
            attn = attn / torch.sum(attn, dim=-2, keepdim=True)
            updates = torch.matmul(attn.transpose(-1, -2), v)                           # Shape: [batch_size, num_heads, num_slots, slot_size // num_heads].
            updates = updates.transpose(1, 2).reshape(B, N_q, -1)                       # Shape: [batch_size, num_slots, slot_size].
            
            # Slot update.
            slots = self.gru(updates.view(-1, self.slot_size),
                             slots_prev.view(-1, self.slot_size))
            slots = slots.view(-1, N_q, self.slot_size)
            slots = slots + self.mlp(self.norm_mlp(slots))
        
        return slots, attn_vis, attn_logits


class SlotAttentionEncoder(nn.Module):
    
    def __init__(self, num_iterations, num_slots,
                 input_channels, slot_size, mlp_hidden_size, pos_channels, truncate='bi-level', init_method='embedding', num_heads = 1, drop_path = 0.0):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.pos_channels = pos_channels
        self.init_method = init_method

        self.layer_norm = nn.LayerNorm(input_channels)
        self.mlp = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.ReLU(),
            linear(input_channels, input_channels))
        
        assert init_method in ['shared_gaussian', 'embedding']
        if init_method == 'shared_gaussian':
            # Parameters for Gaussian init (shared by all slots).
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)
        elif init_method == 'embedding':
            self.slots_init = nn.Embedding(num_slots, slot_size)
            nn.init.xavier_uniform_(self.slots_init.weight)
        else:
            raise NotImplementedError
        
        self.slot_attention = SlotAttention(
            num_iterations,
            input_channels, slot_size, mlp_hidden_size, truncate, num_heads, drop_path=drop_path)
    
    def forward(self, x, previous_slots=None , last_SA = False):
        # `image` has shape: [batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        B, *_ = x.size() # batch size?
        dtype = x.dtype
        device = x.device

        if last_SA:
            x = self.mlp(self.layer_norm(x))
        # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].

        # Slot Attention module.
        init_slots = self.slots_initialization(B, dtype, device, previous_slots)

        slots, attn, attn_logits = self.slot_attention(x, init_slots)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].
        
        return slots, attn, init_slots, attn_logits
    
    def slots_initialization(self, B, dtype, device, previous_slots=None):
        if previous_slots is not None:
            slots_init = previous_slots
        elif self.init_method == 'shared_gaussian':
            slots_init = torch.empty((B, self.num_slots, self.slot_size), dtype=dtype, device=device).normal_()
            slots_init = self.slot_mu + torch.exp(self.slot_log_sigma) * slots_init
        elif self.init_method == 'embedding':
            slots_init = self.slots_init.weight.expand(B, -1, -1).contiguous()
        
        return slots_init


class MultiScaleSlotAttentionEncoder(nn.Module):
    """
        Mutli-Scale Slot Attention Encoder where no weights are shared, i.e. every scale has its own encoder
    """
    
    def __init__(
        self, num_iterations, num_slots, input_channels, slot_size, mlp_hidden_size, pos_channels,
        truncate='bi-level', init_method='embedding', ms_which_encoder_layers = [9, 10, 11], 
        fusion_method = "mean", val_mask_size=320, num_heads=1, drop_path=0.0
    ):
        super().__init__()
        self.it_counter = 0

        # Store hyper-paramters
        self.ms_which_encoder_layers = ms_which_encoder_layers
        self.residual = fusion_method == "residual"
        self.val_mask_size = val_mask_size

        # Create the ensemble of slot attention encoders
        self.slot_attention_encoders = nn.ModuleList([
            SlotAttentionEncoder(
                num_iterations, num_slots, input_channels, slot_size, mlp_hidden_size,
                pos_channels, truncate, init_method, num_heads, drop_path
            ) for i in range(len(ms_which_encoder_layers))
        ])
        
        # Set fusion method according to provided args
        if fusion_method not in FUSION_STRING_MAPPING:
            raise ValueError(f"The provided fusion {fusion_method} method does not exist!")
        self.fusion_module = FUSION_STRING_MAPPING.get(fusion_method)(slot_size, len(ms_which_encoder_layers))

    def align_slots(self, slots_list, attn_list, init_slots_list, attn_logits_list):
        """
        Based on the list of attention maps tensors, we use hungarian matching to ensure they correspond across scales
        and if not, align them and the slots accordingly IN PLACE.

        params:
            shape of slots list: torch.Size([B, 6, 256]), n_scales
            shape of attn list: torch.Size([B, 196, 6]), n_scales
            shape of init slots list: torch.Size([B, 6, 256]), n_scales
            shape of attn logits list: torch.Size([B, 1, 196, 6]), n_scales

        """
        # if there is only one scale, we do not need to align anything
        if len(slots_list) < 2:
            return slots_list, attn_list, init_slots_list, attn_logits_list
        
        def preprocess_attn_mask_batch(input: torch.tensor):
            """
            expected initial shape: [B, H*W, C]
            B = batch size
            H*W = mask size [VOC PASCAL DINOSAUR -> 196]
            C = number of slots [VOC PASCAL DINOSAUR -> 6]
            output shape: [B, C, H*W]
            """
            height, width = np.sqrt(input.shape[1]), np.sqrt(input.shape[1])
            # check if height has no decimals
            if not height.is_integer():
                raise ValueError("Height/width of input is not a square number")
            height = int(height)
            width = int(width)

            input_transposed = input.transpose(-1,-2).reshape(input.shape[0], input.shape[2], height, width) # shape now [B, C, H, W]
            # argmax and squeeze to get slot IDs
            input_slot_id = input_transposed.unsqueeze(2).argmax(1).squeeze(1) # shape now [B, H, W], contains slot IDs
            # one hot encode
            input_one_hot = torch.nn.functional.one_hot(input_slot_id, num_classes=input.shape[2]).to(torch.float32).permute(0,3,1,2) # shape now [B, C, H, W]
            input_one_hot_flattened = input_one_hot.reshape(input_one_hot.shape[0], input_one_hot.shape[1], -1)
            
            if input_one_hot_flattened.shape[1] != input.shape[2] or input_one_hot_flattened.shape[2] != input.shape[1]:
                print(f"original input shape: {input.shape}, transposed: {input_transposed.shape}, slot_id: {input_slot_id.shape}, one_hot: {input_one_hot.shape}, one_hot_flattened: {input_one_hot_flattened.shape}")
                raise ValueError("Preprocessing of input did not work as expected")
            return input_one_hot_flattened
        
        def plot_ordering(previous_attn, current_attn, current_attn_ordered, batch_index, upsample_size, iteration, i):
            """
            expected attn shape [B, 196, 6]). Creates a threefold plot for the ordering of the attention maps, 
            where the first plot shows the previous attention map, the second the current attention map and 
            the third the ordered current attention map by colorcoding the six slots
            """
            if iteration%100!=0:
                return
            colors_6 = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b"   # Brown
            ]
            attn_masks = [previous_attn, current_attn, current_attn_ordered]
            
            attn_masks = [attn_mask[batch_index] for attn_mask in attn_masks]
            h_w = int(math.sqrt(attn_masks[0].shape[0]))
            attn_mask_list_bi = [attn_mask.reshape(h_w, h_w, previous_attn.shape[2]) for attn_mask in attn_masks]
            attn_mask_upsampled = [F.interpolate(
                attn_mask.permute(2, 0, 1).unsqueeze(0), 
                size=(upsample_size, upsample_size),  
                mode='bilinear'
            ).squeeze(0).permute(1, 2, 0)  for attn_mask in attn_mask_list_bi]
            attn_masks_np = [attn_mask.clone().detach().cpu().numpy() for attn_mask in attn_mask_upsampled]

            attn_slot_id = [np.argmax(attn_mask, axis=-1) for attn_mask in attn_masks_np]
           
            fig, axs = plt.subplots(1, len(attn_slot_id), figsize=(24,3))
    
            cmap = mcolors.ListedColormap(colors_6)
            norm = plt.Normalize(vmin=0, vmax=5)
            titles =["Attention map at layer i", "Attention map at layer i+1", "Ordered Attention map at layer i+1"]
            for i, attn in enumerate(attn_slot_id):
                im = axs[i].imshow(attn, cmap=cmap, norm = norm,  interpolation='nearest')
                axs[i].set_title(titles[i])
                axs[i].axis('off') 
            
            fig.tight_layout()
            plt.savefig(f'ordering/plot_odering_{iteration}_{i}.png')
            plt.close() # Close the plot to avoid displaying it

            """previous_attn = previous_attn[batch_index]
            current_attn = current_attn[batch_index]
            current_attn_ordered = current_attn_ordered[batch_index]
         
            previous_attn = previous_attn.clone().detach().cpu().numpy()
            current_attn = current_attn.clone().detach().cpu().numpy()
            current_attn_ordered = current_attn_ordered.clone().detach().cpu().numpy()
            
            # reshape 102400 to 320x320
            previous_attn = previous_attn.reshape(upsample_size, upsample_size, 6)
            current_attn = current_attn.reshape(upsample_size, upsample_size, 6)
            current_attn_ordered = current_attn_ordered.reshape(upsample_size, upsample_size, 6)

            # instead of one hot encoding, transform into slot ids
            current_attn = np.argmax(current_attn, axis=-1)
            current_attn_ordered = np.argmax(current_attn_ordered, axis=-1)
            previous_attn = np.argmax(previous_attn, axis=-1)
            
            #check if current attn is different from current attn ordered
            if np.all(current_attn == current_attn_ordered):
                print("Attention maps are the same after ordering")
                return
            cmap = cm.get_cmap('tab20', 6) 
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            # Plot previous attention map
            print(np.unique(previous_attn), np.unique(current_attn), np.unique(current_attn_ordered))
            im0 = axs[0].imshow(previous_attn, cmap=cmap, interpolation='nearest')
            axs[0].set_title("Previous Attention Map")
            axs[0].axis('off')  # Remove axis ticks
            
            # Plot current attention map
            im1 = axs[1].imshow(current_attn, cmap=cmap, interpolation='nearest')
            axs[1].set_title("Current Attention Map")
            axs[1].axis('off')  # Remove axis ticks
            
            # Plot ordered current attention map
            im2 = axs[2].imshow(current_attn_ordered, cmap=cmap, interpolation='nearest')
            axs[2].set_title("Ordered Current Attention Map")
            axs[2].axis('off')  # Remove axis ticks
            
            # Add a colorbar (legend) that maps color to slot number
            fig.colorbar(im2, ax=axs, orientation='horizontal', fraction=0.02, pad=0.04, ticks=np.arange(6))


            # Save the plot as an image file
            plt.savefig(f'/visinf/home/vilab01/spot/plots/slot_ordering_{iteration}_layer_{layer}.png')  # You can change the file name and extension (e.g., .jpg, .png)
            plt.close()  # Close the plot to avoid displaying it
            """
        def hungarian_matching(attn_masks_1: torch.tensor, attn_masks_2: torch.tensor):
            """
            expected input size [B, C, H*W]
            performs hungarian matching on the inputs and returns the list of matched indices
            """
            batch_size = attn_masks_1.shape[0]        
            slot_orderings = [unsupervised_mask_iou(attn_masks_1[i], attn_masks_2[i], "hungarian", None, 0.0, True) for i in range(batch_size)]
            return slot_orderings
        
        
        for i in range(len(attn_list)-1):
            # in each iteration, we want to order the slots + attention maps at index i+1 
            # based on the slots/attention maps at index i
            attn_previous_scale_copy = attn_list[i].clone()
            attn_this_scale_copy = attn_list[i+1].clone()
            # preprocess input
            attn_masks_previous_scale = preprocess_attn_mask_batch(attn_list[i]) # shape [64, 196, 6]
            attn_masks_this_scale = preprocess_attn_mask_batch(attn_list[i+1]) # shape [64, 196, 6]
            
            #attn_previous_scale_copy = attn_masks_previous_scale.clone()
            #attn_this_scale_copy = attn_masks_this_scale.clone()

            # placeholders for ordered slots and attention maps etc
            slots_ordered = torch.zeros_like(slots_list[i+1]) # shape [64, 6, 256]
            attn_ordered = torch.zeros_like(attn_list[i+1]) # shape [64, 196, 6]
            init_slots_ordered = torch.zeros_like(init_slots_list[i+1]) # shape [64, 6, 256]
            attn_logits_ordered = torch.zeros_like(attn_logits_list[i+1])  # shape [64, 1, 196, 6]

            # perform hungarian matching
            slot_orderings_list_over_batch = hungarian_matching(attn_masks_previous_scale, attn_masks_this_scale)
            
            # get correct order for the second input for each batch
            slot_orderings_list_this_scale = [order[1] for order in slot_orderings_list_over_batch]  
           
            for b, slot_order in enumerate(slot_orderings_list_this_scale):
                # align in the respective slot dimension
                slots_ordered[b] = slots_list[i+1][b, slot_order, :]
                attn_ordered[b] = attn_list[i+1][b, :, slot_order]
                init_slots_ordered[b] = init_slots_list[i+1][b, slot_order, :]
                attn_logits_ordered[b] = attn_logits_list[i+1][b, :, :, slot_order]

            # replace the original slots and attention maps with the ordered ones
            slots_list[i+1] = slots_ordered
            attn_list[i+1] = attn_ordered
            init_slots_list[i+1] = init_slots_ordered
            attn_logits_list[i+1] = attn_logits_ordered
           
            if DO_PLOT_ORDERING:
                plot_ordering(attn_previous_scale_copy, attn_this_scale_copy, attn_ordered, 0, self.val_mask_size, self.it_counter, i)
                self.it_counter += 1
                 # plot if attn_ordered is different 
                 
               # if not torch.all(attn_this_scale_copy == ordered_attn_mask_batch):
                #    ordered_attn_mask_batch = preprocess_attn_mask_batch(attn_list[i+1])
                   # for batch_index in range(attn_this_scale_copy.shape[0]):
                    #    if not torch.all(attn_this_scale_copy[batch_index] == ordered_attn_mask_batch[batch_index]):
                       #     plot_ordering(attn_previous_scale_copy, attn_this_scale_copy, ordered_attn_mask_batch, batch_index, self.val_mask_size, self.counter)
                       #     self.counter += 1
                          #  break   
        return slots_list, attn_list, init_slots_list, attn_logits_list   
     
    def forward(self, x, image=None, save_folder=None):
        # Lists for storing intermediate scale results
        slots_list = []
        attn_list = []
        init_slots_list = []
        attn_logits_list = []
        old_slots = None

        # Enumerate over all SA scales
        for i, (sae, inp) in enumerate(zip(self.slot_attention_encoders, x)):

            # Whether this is the last layer, necessary for mlp / norm at final layer
            is_last_layer = True if i == len(x) - 1 else False

            # Hierarchical slot initialization for residual connection
            if self.residual:
                if i == 0:
                    slots, attn, init_slots, attn_logits = sae(inp, None, is_last_layer)
                    old_slots = slots.clone().detach() # detach as to not compute gradients for hierarchical slot init
                else:
                    slots, attn, init_slots, attn_logits = sae(inp, slots_list[-1], is_last_layer)
                    # residual connection
                    slots = slots + old_slots
                    old_slots = slots.clone().detach()
            
            # Random slot initialization at each layer
            else:
                slots, attn, init_slots, attn_logits = sae(inp, None, is_last_layer)
            
            # Append to lists
            slots_list.append(slots)
            attn_list.append(attn)
            init_slots_list.append(init_slots)
            attn_logits_list.append(attn_logits)
        
        
        # ensure slots correspond across scales
        if not self.residual:
           slots_list, attn_list, init_slots_list, attn_logits_list = self.align_slots(slots_list, attn_list, init_slots_list, attn_logits_list)
        
        
        # Fusion across scales
        agg_slots, agg_attn, agg_init_slots, agg_attn_logits = self.fusion_module(slots_list, attn_list, init_slots_list, attn_logits_list)

        # Visualization of slot attention layers and fused result
        if image is not None:
            visualize_layer_attn(attn_list, image, agg_attn, batch_index = 0, upsample_size=self.val_mask_size, iteration=self.it_counter, n_slots = slots_list[0].shape[1], mode ="distinct", save_folder=save_folder)
            visualize_layer_attn(attn_list, image, agg_attn, batch_index = 0, upsample_size=self.val_mask_size, iteration=self.it_counter, n_slots = slots_list[0].shape[1], mode = "overlay", save_folder = save_folder)
            self.it_counter += 1
        
        # Return results
        return agg_slots, agg_attn, agg_init_slots, agg_attn_logits

      
class MultiScaleSlotAttentionEncoderShared(nn.Module):
    """
        Mutli-Scale Slot Attention Encoder where all weights are shared, i.e. every scale shares the same encoder
    """

    def __init__(self, num_iterations, num_slots,
                 input_channels, slot_size, mlp_hidden_size, pos_channels, truncate='bi-level', init_method='embedding', ms_which_encoder_layers = [9, 10, 11], concat_method = "add", slot_initialization=None, num_heads = 1, drop_path = 0.0):
        super().__init__()
        
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_channels = input_channels
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.pos_channels = pos_channels
        self.init_method = init_method

        self.layer_norm = nn.LayerNorm(input_channels)
        self.mlp = nn.Sequential(
            linear(input_channels, input_channels, weight_init='kaiming'),
            nn.ReLU(),
            linear(input_channels, input_channels))
        
        self.ms_which_encoder_layers = ms_which_encoder_layers
        self.concat_method = concat_method
        assert init_method in ['shared_gaussian', 'embedding']
        if init_method == 'shared_gaussian':
            # Parameters for Gaussian init (shared by all slots).
            self.slot_mu = nn.Parameter(torch.zeros(1, 1, slot_size))
            self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_size))
            nn.init.xavier_uniform_(self.slot_mu)
            nn.init.xavier_uniform_(self.slot_log_sigma)
        elif init_method == 'embedding':
            self.slots_init = nn.Embedding(num_slots, slot_size)
            nn.init.xavier_uniform_(self.slots_init.weight)
        else:
            raise NotImplementedError
        
        self.slot_attention = SlotAttention(
            num_iterations,
            input_channels, slot_size, mlp_hidden_size, truncate, num_heads, drop_path=drop_path)
        
    def forward(self, x):
        # x is a ModuleList?
        # `image` has shape: [n_scales ,batch_size, img_channels, img_height, img_width].
        # `encoder_grid` has shape: [batch_size, pos_channels, enc_height, enc_width].
        
        B, *_ = x[0].size() # batch size?
        dtype = x[0].dtype
        device = x[0].device
        
        ms_slots = []
        ms_attn = []
        ms_attn_logits = []
        ms_init_slots = []

        for i in range(len(self.ms_which_encoder_layers)):
            init_slots = self.slots_initialization(B, dtype, device)
            ms_init_slots.append(init_slots)
            # `slots` has shape: [batch_size, num_slots, slot_size].
            # `attn` has shape: [batch_size, enc_height * enc_width, num_slots].
            x_item = self.mlp(self.layer_norm(x[i]))
           
            # `x` has shape: [batch_size, enc_height * enc_width, cnn_hidden_size].
            slots, attn, attn_logits = self.slot_attention(x_item, init_slots)
            ms_slots.append(slots)
            ms_attn.append(attn)
            ms_attn_logits.append(attn_logits)
        
        return self.concat_slot_attention(self.concat_method, ms_slots, ms_attn, ms_init_slots, ms_attn_logits)
        
    
    def concat_slot_attention(self, concat_method, ms_slots, ms_attn, ms_init_slots, ms_attn_logits):
        if concat_method == "add":
            ms_slots = torch.stack(ms_slots).sum(0)
            # todo: does this make sense?
            ms_attn = torch.stack(ms_attn).sum(0)
            ms_attn_logits = torch.stack(ms_attn_logits).sum(0)
            ms_init_slots = torch.stack(ms_init_slots).sum(0)

        elif concat_method == "mean":
            ms_slots = torch.stack(ms_slots).mean(0)
            ms_attn = torch.stack(ms_attn).mean(0)
            ms_attn_logits = torch.stack(ms_attn_logits).mean(0)
            ms_init_slots = torch.stack(ms_init_slots).mean(0)
        elif concat_method== "residual":
           # TODO: implement!!
            raise NotImplementedError
        else:
            raise NotImplementedError
        
        return ms_slots, ms_attn, ms_init_slots, ms_attn_logits

    def slots_initialization(self, B, dtype, device):
        # The first frame, initialize slots.
        if self.init_method == 'shared_gaussian':
            slots_init = torch.empty((B, self.num_slots, self.slot_size), dtype=dtype, device=device).normal_()
            slots_init = self.slot_mu + torch.exp(self.slot_log_sigma) * slots_init
        elif self.init_method == 'embedding':
            slots_init = self.slots_init.weight.expand(B, -1, -1).contiguous()
        
        return slots_init
