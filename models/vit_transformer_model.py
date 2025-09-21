from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
import torch 
import torch.nn as nn


class VTrans(nn.Module):
    def __init__(self, num_classes,dropout = 0.15, hidden_dim = 1536, model = None):
        super().__init__()
        # self.vit_pretrained = timm.create_model(model_name=model_name, pretrained=True, **timm_kwargs)
        self.vit_pretrained = model
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True, dropout=dropout)
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, frames): # [B, T, 3, 224, 224]
        B, T, C, H, W = frames.shape
        chunk_size = 25   # thay đổi theo VRAM
        embeds_list = []

        for i in range(0, T, chunk_size):
          frames_chunk = frames[:, i:i+chunk_size]  # [B, chunk, 3, H, W]
          frames_chunk = frames_chunk.view(-1, C, H, W)
          with torch.cuda.amp.autocast():
            emb = self.vit_pretrained(frames_chunk)  # [B*chunk, 1536]
          emb = emb.view(B, -1, emb.size(-1))          # [B, chunk, 1536]
          embeds_list.append(emb)

        embeds = torch.cat(embeds_list, dim=1)   # [B, T, 1536]
        out = self.temporal_encoder(embeds)
        out = out.mean(dim=1)
        logits = self.fc(out)
        return logits   
        