from dotenv import load_dotenv
import cv2
import numpy as np
import os
import hydra
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import huggingface_hub
from omegaconf import OmegaConf, open_dict
from logging import WARNING, INFO
import logging
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from utils.datasets_utils import VideoDataset_h5, collate_fn
from sklearn.model_selection import train_test_split
from models.vit_transformer_model import VTrans
from utils.core_utils import train
import torchvision.models as models
from peft import LoraConfig, get_peft_model
from torch.utils.data import Subset

# bật SDPA backend (FlashAttention, mem-efficient, math)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(True)

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
huggingface_hub.login(token=hf_token)


# --- Setup logging ---
logger = logging.getLogger('SmartfileTest')
logger.setLevel(INFO)
NEW_FORMAT = '[%(asctime)s] - [%(levelname)s] - %(message)s'
logger_format = logging.Formatter(NEW_FORMAT)



labels = ['Normal', 'Adenoma', 'Malignant']
label2id = {
    label: idx for idx, label in enumerate(labels)
}
id2labels = {idx: label for label, idx in label2id.items()}

data_paths = {
    'data' : 'datasets/frames/train',
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_data(data_path):
    frames_paths = []
    labels = []
    for label_idx, class_name in id2labels.items():
        class_dir = os.path.join(data_path, class_name)
        for frames_name in os.listdir(class_dir):
            frames_path = os.path.join(class_dir, frames_name)
            frames_paths.append(frames_path)
            labels.append(label_idx)
    return frames_paths, labels


def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
@hydra.main(config_path="configs", version_base=None)
def main(cfg: OmegaConf):
    logger.info("config: %s", cfg)
    seed_torch(int(cfg.seed))
    timm_kwargs = {
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
    model = timm.create_model(model_name= 'hf-hub:MahmoodLab/UNI2-h', pretrained=True, **timm_kwargs)
    # for param in model.parameters():
    #     param.requires_grad = False
    transform = create_transform(**resolve_data_config(model.pretrained_cfg))
    config_lora = LoraConfig(
        r=int(cfg.r),
        lora_alpha=int(cfg.lora_alpha),
        target_modules=cfg.target_modules,
        lora_dropout=float(cfg.lora_dropout),
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    model = get_peft_model(model, config_lora)
    
    #------------Data--------------
    frames_paths, labels = load_data(data_paths['data'])
    train_paths, val_paths, train_labels, val_labels = train_test_split(frames_paths, labels, test_size=0.2)
    
    print("Init Dataset...\n")
    train_data = VideoDataset_h5(train_paths, train_labels, transform=transform)
    val_data = VideoDataset_h5(val_paths, val_labels, transform=transform)
    max_len_video = 1024
    imgs, label = train_data[0] 
    print(f"Frames shape: {imgs.shape}\nLabel: {label}")
    print("Done!")
    
    print("Init DataLoader...\n")
    train_subset = Subset(train_data, range(3))
    val_subset = Subset(val_data, range(3))
    train_loader = DataLoader(train_subset, batch_size = cfg.batch_size, collate_fn = collate_fn, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size = cfg.batch_size, collate_fn = collate_fn, shuffle=False, num_workers=0)
    print("Done!")
    
    #------------Model-------------
    main_model = VTrans(num_classes=int(cfg.num_classes),
                        max_len_video=max_len_video,
                        dropout= float(cfg.dropout), 
                        hidden_dim=int(cfg.hidden_size),
                        model=model)
    
    #-------------Train------------
    train(cfg, train_loader, val_loader, device, main_model)

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main() 
   
    
    
    