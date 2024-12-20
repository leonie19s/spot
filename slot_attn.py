''' Based on SLATE and BOQSA libraries:
https://github.com/singhgautam/slate/blob/master/slot_attn.py
https://github.com/YuLiu-LY/BO-QSA/blob/main/models/slot_attn.py
'''

from utils_spot import *
from timm.models.layers import DropPath

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

    def __init__(self, num_iterations, num_slots, input_channels, slot_size, mlp_hidden_size, pos_channels,
                  truncate='bi-level', init_method='embedding', ms_which_encoder_layers = [9, 10, 11], concat_method = "sum", slot_initialization=None, num_heads = 1, drop_path = 0.0):
        super().__init__()
        
        self.ms_which_encoder_layers = ms_which_encoder_layers
        self.slot_attention_encoders = nn.ModuleList([
            SlotAttentionEncoder(
                num_iterations, num_slots, input_channels, slot_size, mlp_hidden_size,
                pos_channels, truncate, init_method, num_heads, drop_path
            ) for i in range(len(ms_which_encoder_layers))
        ])
        self.slot_initialization = slot_initialization
        # Set aggregation function according to provided args, default to mean
        self.agg_fct = torch.sum
        if concat_method == "mean":
            self.agg_fct = torch.mean
        elif concat_method == "max":
            self.agg_fct = lambda x, dim: torch.max(x, dim = dim).values
        elif concat_method == "None" or concat_method == None:
            # then we only want the last slot from the list
            self.agg_fct = lambda x, dim: x[-1]
        elif concat_method != "sum":
            print(f"Provided aggregation function {concat_method} does not exist, defaulting to sum")
    
    def forward(self, x):

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

            # Hierarchical slot initialization
            if self.slot_initialization == "hierarchical":
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
    
        # Aggregation across scales
        agg_slots = self.agg_fct(torch.stack(slots_list), dim=0)
        agg_attn = self.agg_fct(torch.stack(attn_list), dim =0)
        agg_init_slots = self.agg_fct(torch.stack(init_slots_list), dim=0)
        agg_attn_logits = self.agg_fct(torch.stack(attn_logits_list), dim=0)

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
