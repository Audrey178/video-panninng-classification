from dotenv import load_dotenv
import cv2
import numpy as np
import os
import hydra
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
from utils.datasets_utils import VideoDataset_h5
from sklearn.model_selection import train_test_split
from models.vit_transformer_model import VTrans
from utils.core_utils import train
import torchvision.models as models

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
# huggingface_hub.login(token=hf_token)

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
    'train' : 'datasets/frames/train',
    'val' : 'datasets/frames/validation'
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
    

@hydra.main(config_path='configs', version_base=None)
def main(cfg: OmegaConf):
    logger.info("config: %s", cfg)
    # timm_kwargs = {
    #     'img_size': 224, 
    #     'patch_size': 16, 
    #     'depth': 24,
    #     'num_heads': 24,
    #     'init_values': 1e-5, 
    #     'embed_dim': 1536,
    #     'mlp_ratio': 2.66667*2,
    #     'num_classes': 0, 
    #     'no_embed_class': True,
    #     'mlp_layer': timm.layers.SwiGLUPacked, 
    #     'act_layer': torch.nn.SiLU, 
    #     'reg_tokens': 8, 
    #     'dynamic_img_size': True
    # }
    # model = timm.create_model(model_name= 'hf-hub:MahmoodLab/UNI2-h', pretrained=True, **timm_kwargs)
    # for param in model.parameters():
    #     param.requires_grad = False
    # transform = create_transform(**resolve_data_config(model.pretrained_cfg))
    # Load pretrained ResNet18
    resnet18 = models.resnet18(pretrained=True)

    modules = list(resnet18.children())[:-1]  
    model = torch.nn.Sequential(*modules)
    
    #------------Data-----------
    frames_paths, labels = load_data(data_paths['train'])
    train_paths, val_paths, train_labels, val_labels = train_test_split(frames_paths, labels, test_size=0.2)
    train_data = VideoDataset_h5(train_paths, train_labels)
    val_data = VideoDataset_h5(val_paths, val_labels)
    
    train_loader = DataLoader(train_data, batch_size = 1, shuffle=True, num_workers=10)
    val_loader = DataLoader(val_data, batch_size = 1, shuffle=False, num_workers=10)
    
    #------------Model-------------
    main_model = VTrans(num_classes=int(cfg.num_classes),
                        dropout= float(cfg.dropout), 
                        hidden_dim=int(cfg.hidden_size),
                        model=model)
    
    #-------------Train------------
    train(cfg, train_loader, val_loader, device, main_model)

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main() 
   
    
    
    