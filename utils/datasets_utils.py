from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
import h5py

class VideoDataset(Dataset):
    def __init__(self, frames_dir, labels, img_size = 224, transform = None):
        super().__init__()
        self.frames_paths = [os.path.join(frames_dir, dir) for dir in os.listdir(frames_dir)]
        self.labels = labels
        self.frame_chunks = []
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]) if transform is None else transform

    def __len__(self):
        return len(self.frames_paths)

    def __getitem__(self, index):
        lst_paths = self.frames_paths[index]
        frames = [Image.open(os.path.join(lst_paths, img_file)).convert("RGB") for img_file in os.listdir(lst_paths)]
        frames_transformed = [self.transform(frame) for frame in frames]
        frame_tensor = torch.stack(frames_transformed)
        return frame_tensor, torch.tensor(self.labels[index])
    
    
class VideoDataset_h5(Dataset):
    def __init__(self, h5_file_paths, labels, img_size = 224, transform = None):
        super().__init__()
        self.file_paths = h5_file_paths
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]) if transform is None else transform
        self.labels = labels
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            with h5py.File(file_path, 'r') as f:
                imgs = torch.from_numpy(f['frames'][:]).permute(0, 3, 1, 2).float() / 255.0
            if self.transform is not None:
                imgs = self.transform(imgs)
            return imgs, torch.tensor(label)
        except Exception as e:
            raise ValueError(f'Error: {e}')
        
def collate_fn(batch):
    videos, labels = zip(*batch)  # tuple of (Ti, 3, H, W)
    lengths = [v.shape[0] for v in videos]

    # Pad thành [B, T_max, 3, H, W]
    videos = torch.nn.utils.rnn.pad_sequence(videos, batch_first=True)

    # Mask: [B, T_max], 1=real, 0=pad
    mask = torch.zeros(videos.size(0), videos.size(1), dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = 1

    return videos, torch.tensor(labels), mask

    
 
        
    