import os
from utils.video_utils import extract_and_save

FRAME_STRIDE = 5           
LAP_VAR_THRESH = 100      
TISSUE_THRESH = 0.05  

SAVE_DIR = 'datasets/frames'

data_paths = {
    'train' : 'datasets/videos/train',
    'val' : 'datasets/videos/validation'
}


labels = ['Normal', 'Adenoma', 'Malignant']
label2id = {
    label: idx for idx, label in enumerate(labels)
}
id2labels = {idx: label for label, idx in label2id.items()}

def load_data(data_path, type):
    save_dir = os.path.join(SAVE_DIR, type)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    for label_idx, label_name in id2labels.items():
        class_dir = os.path.join(data_path, label_name)
        for video_file in os.listdir(class_dir):
            video_path = os.path.join(class_dir, video_file)
            video_name = video_file.split('.avi')[0]
            out_dir = os.path.join(save_dir, label_name)
            file_size = extract_and_save(video_path, out_dir, FRAME_STRIDE , LAP_VAR_THRESH, TISSUE_THRESH, label_name, video_name)
            print(file_size)
            
load_data(data_paths['train'], 'train')