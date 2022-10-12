import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

from config import *
## Dataset
def csv_mapping():
    dic = {}
    df = pd.read_csv('tag_locCoor.csv', encoding='big5')
    for i, r in df.iterrows():
        key = r['Img'].replace('.jpg', '')
        dic[key] = {
            'x': r['target_x'],
            'y': r['target_y'],
            'town': r['TOWNNAME']
        }
    
    return dic


class CropsDataset(Dataset):
    def __init__(self,mode):
        self.mode = mode
        self.img_size = (380, 380)
        if self.mode == 'train':
            labels_csv = pd.read_csv(f'{mode}.csv')
        elif self.mode == 'valid':
            labels_csv = pd.read_csv(f'{mode}.csv')
        self.filenames = labels_csv['img_path']
        self.labels = labels_csv['label']
        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomRotation((-15,15)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.RandomAutocontrast(),
                transforms.RandomAdjustSharpness(sharpness_factor=0, p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        
        self.img_info = csv_mapping()
        
    def __getitem__(self, idx):
        img = Image.open(f'{DATA_ROOT}{self.filenames[idx]}').convert('RGB')
        img_name = self.filenames[idx].replace('.jpg', '').split('/')[-1]
        data = self.img_info[img_name]
        data.update({
            "img":self.transforms(img)
        })
        # => (data, label)
        return data, self.labels[idx]
    
    def __len__(self):
        return len(self.labels)
