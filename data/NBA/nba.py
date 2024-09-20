import os
import os.path as osp
import numpy as np
import torch
from pycocotools.coco import COCO
from utils.preprocessing import load_img, process_bbox, augmentation
from torch.utils.data import Dataset

class NBA(Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.data_path = osp.join(cfg.data_dir, 'NBA', 'data')  # Adjust this path to your dataset
        self.datalist = self.load_data()

    def load_data(self):
        datalist = []
        db = COCO(osp.join(self.data_path, 'NBA.json'))  # Adjust the JSON file to your annotations
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_shape = (img['height'], img['width'])
            img_path = osp.join(self.data_path, img['file_name'])

            # Load bounding box or other keypoints specific to your dataset
            bbox = ann['bbox']  
            bbox = process_bbox(bbox, img['width'], img['height'], ratio=getattr(cfg, 'bbox_ratio', 1.25))
            if bbox is None:
                continue

            data_dict = {'img_path': img_path, 'img_shape': img_shape, 'bbox': bbox}
            datalist.append(data_dict)

        return datalist

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # Load image
        img = load_img(img_path)

        # Apply augmentation and transformation
        img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
        img = self.transform(img.astype(np.float32)) / 255.

        inputs = {'img': img}
        targets = {'bbox': bbox}  # Add more target information if needed
        meta_info = {'img_path': img_path}

        return inputs, targets, meta_info
