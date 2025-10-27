import os
import pandas as pd
from torch.utils.data import Dataset

from PIL import Image


class MetaShiftDataset(Dataset):
    def __init__(self, root_dir, stage: str = 'train'):
        self.root_dir = root_dir
        self.stage = stage

        self.imageID_to_group_pkl = pd.read_pickle(os.path.join(self.root_dir, 'imageID_to_group.pkl'))
        self.data, self.labels = self._load_data()

    def _load_data(self):
        if self.stage == 'train':
                data_folder = os.path.join(self.root_dir, 'train')
        elif self.stage == 'val':
                data_folder = os.path.join(self.root_dir, 'val_out_of_domain')
        else:
                raise ValueError(f"Invalid stage: {self.stage}")
        
        classes = os.listdir(data_folder)
        # Get the list of images in the data folder
        images = []
        labels = []

        for c in classes:
            class_folder = os.path.join(data_folder, c)
            for f in os.listdir(class_folder):
                if f.endswith('.jpg') or f.endswith('.png'):
                    images.append(os.path.join(class_folder, f))
                    labels.append(c)

        return images, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image_id = img_path.split('/')[-1].split('.')[0]

        group = self.imageID_to_group_pkl[image_id][0].split('(')[-1].split(')')[0].strip()
        image = Image.open(img_path)
        label = self.labels[idx]

        item = {
            'image': image,
            'label': label,
            'group': group,
            'image_path': img_path,
        }

        return item
