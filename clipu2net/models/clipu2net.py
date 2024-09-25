from collections import OrderedDict
from typing import Tuple
import math

import torch
from torch import nn

from .clip import clip, tokenize
from .clip.model import LayerNorm
from .mini_u2net import MiniU2Net

class Transformer(nn.Module):
    """Pre-normalized Transformer encoder layer.
    """
    def __init__(
        self, 
        d_model: int, 
        n_head: int,
        dropout: float = 0.1,
        masked: bool = True,
        ):
        super().__init__()
        self.masked = masked

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("dropout", nn.Dropout(dropout)),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
            ("dropout2", nn.Dropout(dropout)),
        ]))
        self.ln_2 = LayerNorm(d_model)

    def build_attention_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Masked attention fusion.
        """
        N = x.shape[0]
        src_mask = torch.full((N, N), fill_value=1, dtype=torch.float32, device=x.device)
        diag = torch.eye(N, dtype=torch.float32, device=x.device)
        diag[1:, 0:1] = 1
        src_mask -= diag
        src_mask = src_mask.masked_fill(src_mask == 1, float('-inf'))
        return src_mask

    def attention(self, x: torch.Tensor):
        attn_mask = self.build_attention_mask(x) if self.masked else None
        x = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        return self.dropout1(x)

    def forward(self, x: torch.Tensor):
        # NOTE: Return attention weights here
        x_attn = self.attention(self.ln_1(x))
        x = x + x_attn
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPU2Net(nn.Module):
    """Segmentation from Prompt with a transformer decoder.
    - Use text embedding as query for attention block.
    - Side outputs from attention blocks.
    """
    def __init__(
        self,
        name: str = "ViT-B/16",
        d_model: int = 64,
        num_heads: int = 8,
        input_resolution: int = 336,
        n_layers_multimodal: int = 8,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        ):
        super().__init__()
        self.name = name
        self.d_model = d_model
        self.num_heads = num_heads
        self.n_layers_multimodal = n_layers_multimodal
        self.input_resolution = input_resolution
        self.device = device
        self.extract_layers = [2, 5, 8]
        
        # CLIP
        self.clip, _ = clip.load(name, device=device, jit=False)
        self.clip.visual.interpolate_pos_embed(input_resolution, input_resolution)
        for p in self.clip.parameters():
            p.requires_grad_(False)

        # Projections & Transformer blocks
        self.proj = nn.ModuleDict()
        self.blocks = nn.ModuleDict()
        for i in range(len(self.extract_layers)):
            self.proj[f"x{i+1}"] = nn.Linear(self.clip.vision_width, d_model, bias=False)

        # Projection_z
        # FIXME: Use PACL's projection?
        self.proj["x_v"] = nn.Linear(self.clip.vision_width, d_model, bias=False)
        self.proj["x_z"] = nn.Linear(self.clip.text_width, d_model, bias=False)

        # Masked self-attention as multimodal learning baseline
        self.multimodal = nn.Sequential(*[
            Transformer(d_model=d_model, n_head=num_heads) for _ in range(n_layers_multimodal)
            ])
        # print("[Multimodal] Masked transformer.")

        # Pixel decoder w/ Low&High Resolution outputs
        for i in range(len(self.extract_layers)):
            self.blocks[f"d{i+1}"] = Transformer(d_model=d_model, n_head=num_heads, masked=False)
        self.blocks["d4"] = Transformer(d_model=d_model, n_head=num_heads, masked=False)
        self.saliency = MiniU2Net(
            d_model=d_model, n_channels=64, 
            vision_patch_size=self.clip.visual.patch_size,
            )

    def proj_feat(self, x):
        L, B, C = x.shape
        R = int(math.sqrt(L - 1))
        
        # Remove [CLS]
        x = x[1:, :, :]

        # To 2D feature maps
        x = x.view(R, R, B, C)
        x = x.permute(2, 3, 0, 1).contiguous()
        return x
    
    def tokenize(self, phrases: Tuple[str]) -> torch.Tensor:
        """Wrapper to tokenize phrases.
        """
        return tokenize(phrases).to(self.device)
    
    def encode_text(self, phrases: Tuple[str]) -> torch.Tensor:
        """To save some computational resources.
        """
        phrases = self.tokenize(phrases)
        # x_z = self.clip.encode_text(phrases, return_features=True)
        # x_z = x_z[phrases.argmax(dim=-1), torch.arange(phrases.shape[0])]     # Pool <EOT>
        
        x_z = self.clip.encode_text(phrases, return_features=False)     # Use the full embedding
        return x_z

    def forward(
        self,
        imgs,
        x_z,
        ):
        # Forward CLIP
        self.clip.eval()
        
        # Visual forward
        x_v, [x1, x2, x3] = self.clip.encode_image(imgs, self.extract_layers, return_features=True)

        # Text forward
        if not isinstance(x_z, torch.Tensor):
            x_z = self.encode_text(x_z)

        # Reduction
        x1 = self.proj["x1"](x1)
        x2 = self.proj["x2"](x2)
        x3 = self.proj["x3"](x3)
            
        x_z = self.proj["x_z"](x_z)
        x_v = self.proj["x_v"](x_v)

        # Multimodal correlation
        # Only calculate the correlation between token_z and img_tokens
        # and the self-correlation of img_token_i
        x = torch.cat([x_z.unsqueeze(0), x_v[1:, :, :]], dim=0)
        x = self.multimodal(x)
        
        # U-shaped decoder w/ U^2Net
        d4 = self.blocks["d4"](x)   # No skip connection here (empricially lower performance)
        d3 = self.blocks["d3"](d4 + x3)
        d2 = self.blocks["d2"](d3 + x2)
        d1 = self.blocks["d1"](d2 + x1)
        
        # Saliency map
        skip = [
            self.proj_feat(d1),
            self.proj_feat(d2),
            self.proj_feat(d3),
            self.proj_feat(d4),
        ]
        outputs = self.saliency(imgs, skip)
                    
        return outputs