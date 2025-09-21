import torch
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config


timm_kwargs = {
        'pretrained': True,
        'img_size': 224, 
        'patch_size': 16, 
        'depth': 24,
        'num_heads': 24,
        'init_values': 1e-5, 
        'embed_dim': 1536,
        'mlp_ratio': 2.66667*2,
        'num_classes': 0, 
        'no_embed_class': True,
        'mlp_layer': timm.layers.SwiGLUPacked, 
        'act_layer': torch.nn.SiLU, 
        'reg_tokens': 8, 
        'dynamic_img_size': True
  }
 

class TimmViTEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'hf-hub:MahmoodLab/UNI2-h', 
                 kwargs:dict = timm_kwargs, 
                 pool: bool = True):
        super().__init__()
        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        try: 
            out = self.model(x)
            if isinstance(out, list):
                assert len(out) == 1
                out = out[0]
            if self.pool:
                out = self.pool(out).squeeze(-1).squeeze(-1)
                return out
        except Exception as e:
            raise ValueError(f"Error at ViT: {e}")
        