from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import torch 
import torch.nn as nn


class VTrans(nn.Module):
    def __init__(self, num_classes, max_len_video, dropout = 0.15, hidden_dim = 1536, model = None):
        super().__init__()
        # self.vit_pretrained = timm.create_model(model_name=model_name, pretrained=True, **timm_kwargs)
        self.vit_pretrained = model
        self.vit_pretrained.set_grad_checkpointing(True)
        
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.max_len_video = max_len_video
        
        self.dropout = dropout
        self.pos_embeds = nn.Embedding(max_len_video, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=dropout,  norm_first=True)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, frames, mask=None):  # [T, 3, 224, 224]
      B, T, C, H, W = frames.shape
      frames = frames.view(-1, C, H, W)          # [B*T, 3, H, W]
      with torch.cuda.amp.autocast(dtype=torch.float16):
        embeds = self.vit_pretrained.forward_features(frames)  # [B*T, hidden_dim]
    
      # reshape lại thành [B, T, hidden_dim]
      embeds = embeds.view(B, T, -1)  
      pos_ids = torch.arange(T, device=frames.device).unsqueeze(0).expand(B, T)  # [B, T]
      embeds = embeds + self.pos_embeds(pos_ids)
      
      out = self.temporal_encoder(embeds, src_key_padding_mask=~mask)  # [B, T, hidden_dim]
      
      # Temporal pooling: chỉ tính trung bình trên frame thực
      if mask is not None:
          lengths = mask.sum(dim=1, keepdim=True)  # [B, 1]
          out = (out * mask.unsqueeze(-1)).sum(dim=1) / lengths  # masked mean
      else:
          out = out.mean(dim=1)

      # Classification
      logits = self.fc(out)  # [B, num_classes]
      return logits
        