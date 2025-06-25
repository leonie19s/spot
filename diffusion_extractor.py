from PIL import Image
import torch
import numpy as np
import os
from PIL import Image
import PIL
import torch
from diffusers import (
    AutoencoderKL, 
    UNet2DConditionModel, 
    DDIMScheduler, 
    StableDiffusionPipeline
)
from transformers import (
    CLIPModel, 
    CLIPTextModel, 
    CLIPTokenizer
)
from diffusers import DDIMScheduler




class DiffusionExtractor:
    """
    Module for running either the generation or inversion process 
    and extracting intermediate feature maps.
    """
    def __init__(self, config, device):
        self.device = device
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
        )
        self.num_timesteps = config["num_timesteps"]
        self.scheduler.set_timesteps(self.num_timesteps)
        self.generator = torch.Generator(self.device).manual_seed(config.get("seed", 0))
        self.batch_size = config.get("batch_size", 1)

        self.unet, self.vae, self.clip, self.clip_tokenizer = init_models(device=self.device, model_id=config["model_id"])
        self.prompt = config.get("prompt", "")
        self.negative_prompt = config.get("negative_prompt", "")
        self.change_cond(self.prompt, "cond")
        self.change_cond(self.negative_prompt, "uncond")
        
        self.diffusion_mode = config.get("diffusion_mode", "generation")
        if "idxs" in config and config["idxs"] is not None:
            self.idxs = config["idxs"]
        else:
            self.idxs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.output_resolution = config["output_resolution"]

        # Note that save_timestep is in terms of number of generation steps
        # save_timestep = 0 is noise, save_timestep = T is a clean image
        # generation saves as [0...T], inversion saves as [T...0]
        self.save_timestep = config.get("save_timestep", [])

        print(f"diffusion_mode: {self.diffusion_mode}")
        print(f"idxs: {self.idxs}")
        print(f"output_resolution: {self.output_resolution}")
        print(f"prompt: {self.prompt}")
        print(f"negative_prompt: {self.negative_prompt}")

    def change_cond(self, prompt, cond_type="cond"):
        with torch.no_grad():
            with torch.autocast("cuda"):
                _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompt)
                new_cond = new_cond.expand((self.batch_size, *new_cond.shape[1:]))
                new_cond = new_cond.to(self.device)
                if cond_type == "cond":
                    self.cond = new_cond
                    self.prompt = prompt
                elif cond_type == "uncond":
                    self.uncond = new_cond
                    self.negative_prompt = prompt
                else:
                    raise NotImplementedError

    def run_generation(self, latent, guidance_scale=-1, min_i=None, max_i=None):
        xs = generalized_steps(
            latent,
            self.unet, 
            self.scheduler, 
            run_inversion=False, 
            guidance_scale=guidance_scale, 
            conditional=self.cond, 
            unconditional=self.uncond, 
            min_i=min_i,
            max_i=max_i
        )
        return xs
    
    def run_inversion(self, latent, guidance_scale=-1, min_i=None, max_i=None):
        xs = generalized_steps(
            latent, 
            self.unet, 
            self.scheduler, 
            run_inversion=True, 
            guidance_scale=guidance_scale, 
            conditional=self.cond, 
            unconditional=self.uncond,
            min_i=min_i,
            max_i=max_i
        )
        return xs

    def get_feats(self, latents, extractor_fn, preview_mode=False):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        outputs = extractor_fn(latents)
        if not preview_mode:
            feats = []
            for timestep in self.save_timestep:
                timestep_feats = collect_and_resize_feats(self.unet, self.idxs, timestep, self.output_resolution)
                feats.append(timestep_feats)
            feats = torch.stack(feats, dim=1)
            init_resnet_func(self.unet, reset=True)
        else:
            feats = None
        return feats, outputs

    def latents_to_images(self, latents):
        latents = latents.to(self.device)
        latents = latents / 0.18215
        images = self.vae.decode(latents.to(self.vae.dtype)).sample
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        return [Image.fromarray(image) for image in images]

    def forward(self, images=None, latents=None, guidance_scale=-1, preview_mode=False):
        if images is None:
            if latents is None:
                latents = torch.randn((self.batch_size, self.unet.in_channels, 512 // 8, 512 // 8), device=self.device, generator=self.generator)
            if self.diffusion_mode == "generation":
                if preview_mode:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale, max_i=self.end_timestep)
                else:
                    extractor_fn = lambda latents: self.run_generation(latents, guidance_scale)
            elif self.diffusion_mode == "inversion":
                raise NotImplementedError
        else:
            images = torch.nn.functional.interpolate(images, size=512, mode="bilinear").half()
            latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215
            if self.diffusion_mode == "inversion":
                extractor_fn = lambda latents: self.run_inversion(latents, guidance_scale)
            elif self.diffusion_mode == "generation":
                raise NotImplementedError
        
        with torch.no_grad():
            with torch.autocast("cuda"):
                return self.get_feats(latents, extractor_fn, preview_mode=preview_mode)



def get_tokens_embedding(clip_tokenizer, clip, device, prompt):
  tokens = clip_tokenizer(
    prompt,
    padding="max_length",
    max_length=clip_tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
    return_overflowing_tokens=True,
  )
  input_ids = tokens.input_ids.to(device)
  embedding = clip(input_ids).last_hidden_state
  return tokens, embedding

def latent_to_image(vae, latent):
  latent = latent / 0.18215
  image = vae.decode(latent.to(vae.dtype)).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  image = (image[0] * 255).round().astype("uint8")
  image = Image.fromarray(image)
  return image

def image_to_latent(vae, image, generator=None, mult=64, w=512, h=512):
  image = image.resize((w, h), resample=PIL.Image.LANCZOS)
  image = np.array(image).astype(np.float32)
  # remove alpha channel
  if len(image.shape) == 2:
    image = image[:, :, None]
  else:
    image = image[:, :, (0, 1, 2)]
  # (b, c, w, h)
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)
  image = image / 255.0
  image = 2. * image - 1.
  image = image.to(vae.device)
  image = image.to(vae.dtype)
  return vae.encode(image).latent_dist.sample(generator=generator) * 0.18215

def get_xt_next(xt, et, at, at_next, eta):
  """
  Uses the DDIM formulation for sampling xt_next
  Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
  """
  x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
  if eta == 0:
    c1 = 0
  else:
    c1 = (
      eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    )
  c2 = ((1 - at_next) - c1 ** 2).sqrt()
  xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(et) + c2 * et
  return x0_t, xt_next

def generalized_steps(x, model, scheduler, **kwargs):
  """
  Performs either the generation or inversion diffusion process.
  """
  seq = scheduler.timesteps
  seq = torch.flip(seq, dims=(0,))
  b = scheduler.betas
  b = b.to(x.device)

  with torch.no_grad():
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    if kwargs.get("run_inversion", False):
      seq_iter = seq_next
      seq_next_iter = seq
    else:
      seq_iter = reversed(seq)
      seq_next_iter = reversed(seq_next)

    x0_preds = [x]
    xs = [x]
    for i, (t, next_t) in enumerate(zip(seq_iter, seq_next_iter)):
      max_i = kwargs.get("max_i", None)
      min_i = kwargs.get("min_i", None)
      if max_i is not None and i >= max_i:
        break
      if min_i is not None and i < min_i:
        continue
      
      t = (torch.ones(n) * t).to(x.device)
      next_t = (torch.ones(n) * next_t).to(x.device)
      if t.sum() == -t.shape[0]:
        at = torch.ones_like(t)
      else:
        at = (1 - b).cumprod(dim=0).index_select(0, t.long())
      if next_t.sum() == -next_t.shape[0]:
        at_next = torch.ones_like(t)
      else:
        at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
      
      # Expand to the correct dim
      at, at_next = at[:, None, None, None], at_next[:, None, None, None]

      if kwargs.get("run_inversion", False):
        set_timestep(model, len(seq_iter) - i - 1)
      else:
        set_timestep(model, i)

      xt = xs[-1].to(x.device)
      cond = kwargs["conditional"]
      
      
      guidance_scale = kwargs.get("guidance_scale", -1)
      if guidance_scale == -1:
        et = model(xt, t, encoder_hidden_states=cond).sample
      else:
        # If using Classifier-Free Guidance, the saved feature maps
        # will be from the last call to the model, the conditional prediction
        uncond = kwargs["unconditional"]
        et_uncond = model(xt, t, encoder_hidden_states=uncond).sample
        et_cond = model(xt, t, encoder_hidden_states=cond).sample
        et = et_uncond + guidance_scale * (et_cond - et_uncond)
      
      eta = kwargs.get("eta", 0.0)
      x0_t, xt_next = get_xt_next(xt, et, at, at_next, eta)

      x0_preds.append(x0_t)
      xs.append(xt_next.to('cpu'))

    return x0_preds

def freeze_weights(weights):
  for param in weights.parameters():
    param.requires_grad = False

def init_models(
    device="cuda",
    model_id="runwayml/stable-diffusion-v1-5",
    freeze=True
  ):
  # Set model weights to mirror since
  # runwayml took down the weights for SDv1-5
  if model_id == "runwayml/stable-diffusion-v1-5":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16, # changed this from float16
  )
  unet = pipe.unet
  vae = pipe.vae
  clip = pipe.text_encoder
  clip_tokenizer = pipe.tokenizer
  unet.to(device)
  vae.to(device)
  clip.to(device)
  if freeze:
    freeze_weights(unet)
    freeze_weights(vae)
    freeze_weights(clip)
  return unet, vae, clip, clip_tokenizer

def collect_and_resize_feats(unet, idxs, timestep, resolution=-1):
  latent_feats = collect_feats(unet, idxs=idxs)
  latent_feats = [feat[timestep] for feat in latent_feats]
  if resolution > 0:
      latent_feats = [torch.nn.functional.interpolate(latent_feat, size=resolution, mode="bilinear") for latent_feat in latent_feats]
  latent_feats = torch.cat(latent_feats, dim=1)
  return latent_feats


def init_resnet_func(
  unet,
  save_hidden=False,
  use_hidden=False,
  reset=True,
  save_timestep=[],
  idxs=[(1, 0)]
):
  def new_forward(self, input_tensor, temb):
    # https://github.com/huggingface/diffusers/blob/ad9d7ce4763f8fb2a9e620bff017830c26086c36/src/diffusers/models/resnet.py#L372
    hidden_states = input_tensor

    hidden_states = self.norm1(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    if self.upsample is not None:
      input_tensor = self.upsample(input_tensor)
      hidden_states = self.upsample(hidden_states)
    elif self.downsample is not None:
      input_tensor = self.downsample(input_tensor)
      hidden_states = self.downsample(hidden_states)

    hidden_states = self.conv1(hidden_states)

    if temb is not None:
      temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]
      hidden_states = hidden_states + temb

    hidden_states = self.norm2(hidden_states)
    hidden_states = self.nonlinearity(hidden_states)

    hidden_states = self.dropout(hidden_states)
    hidden_states = self.conv2(hidden_states)

    if self.conv_shortcut is not None:
      input_tensor = self.conv_shortcut(input_tensor)

    if save_hidden:
      if save_timestep is None or self.timestep in save_timestep:
        self.feats[self.timestep] = hidden_states
    elif use_hidden:
      hidden_states = self.feats[self.timestep]
    output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
    return output_tensor
  
  layers = collect_layers(unet, idxs)
  for module in layers:
    module.forward = new_forward.__get__(module, type(module))
    if reset:
      module.feats = {}
      module.timestep = None

def set_timestep(unet, timestep=None):
  for name, module in unet.named_modules():
    module_name = type(module).__name__
    module.timestep = timestep

def collect_layers(unet, idxs=None):
  layers = []
  for i, up_block in enumerate(unet.up_blocks):
    for j, module in enumerate(up_block.resnets):
      if idxs is None or (i, j) in idxs:
        layers.append(module)
  return layers

def collect_dims(unet, idxs=None):
  dims = []
  for i, up_block in enumerate(unet.up_blocks):
      for j, module in enumerate(up_block.resnets):
          if idxs is None or (i, j) in idxs:
            dims.append(module.time_emb_proj.out_features)
  return dims

def collect_feats(unet, idxs):
  feats = []
  layers = collect_layers(unet, idxs)
  for module in layers:
    feats.append(module.feats)
  return feats

def set_feats(unet, feats, idxs):
  layers = collect_layers(unet, idxs)
  for i, module in enumerate(layers):
    module.feats = feats[i]