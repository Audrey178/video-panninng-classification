import os
import h5py
import numpy as np 
from PIL import Image
import cv2


def initialize_hdf5_bag(first_patch):
    save_path, name, img_patch, label = tuple(first_patch.values())
    file_path = os.path.join(save_path, name) + '.h5'
    file = h5py.File(file_path, 'w')
    img_patch = np.array(img_patch)[np.newaxis, ...]
    dtype = img_patch.dtype
    
    img_shape=img_patch.shape
    max_shape = (None, ) + img_shape[1:]
    ds = file.create_dataset('frames', shape=img_shape, maxshape=max_shape, chunks=img_shape, dtype=dtype)
    ds[:] = img_patch
    ds.attrs['video_name'] = name
    ds.attrs['label'] = label
    
    file.close()

def savePatchIter_bag_hdf5(patch):
    save_path, name, img_patch, _ = tuple(patch.values())
    img_patch = np.array(img_patch)[np.newaxis,...]
    img_shape = img_patch.shape

    file_path = os.path.join(save_path, name)+'.h5'
    file = h5py.File(file_path, "a")

    dset = file['frames']
    dset.resize(len(dset) + img_shape[0], axis=0)
    dset[-img_shape[0]:] = img_patch

    file.close()
    
def is_blurry(img_bgr, lap_thresh=100):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < lap_thresh

def tissue_fraction(img_brg):
    hsv = cv2.cvtColor(img_brg, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2]
    _, th = cv2.threshold(v, 0, 255 , cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th > 0).sum()/th.size

def extract_and_save(video_path, out_dir, stride, lap_thresh, tissue_thres, label ,video_name):
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    all_results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % stride == 0:
            if not is_blurry(frame, lap_thresh) and tissue_fraction(frame) > tissue_thres:
                patch = {
                    'save_path': out_dir,
                    'name': video_name,
                    'img_patch' : frame,
                    'label': label
                }
                if frame_idx == 0:
                    initialize_hdf5_bag(patch)
                else:
                    savePatchIter_bag_hdf5(patch)     
                all_results.append(frame)           
        frame_idx+=1
    
    return len(all_results)