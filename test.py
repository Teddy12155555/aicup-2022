import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import os
import pandas
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from efficientnet_pytorch import EfficientNet
import numpy as np
from tqdm.auto import tqdm
import random

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    # Cuda
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(8863)

data_path = './dataset/'
output_path = './dataset/pred_eff_weight_new_transform_sigmoid.csv'
model_path = './models/Efficient_b4_weight_random_380_rotate30_colorjitter09_newvalid.ckpt'

## Dataset 
class HAMDataset(Dataset):
    def __init__(self, data_path, mode, img_size):
        self.mode = mode
        self.data_path = f'{data_path}{mode}/'
        self.filenames = os.listdir(self.data_path)
        if self.mode == 'train':
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation((-30,30)),
                transforms.Resize(img_size),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
            ])
    def __getitem__(self, idx):
        img = Image.open(f'{self.data_path}{self.filenames[idx]}').convert('RGB')
        if self.mode == 'train':
            return self.transforms(img)
        else:
            return self.transforms(img), self.filenames[idx]
    
    def __len__(self):
        return len(self.filenames)

BATCH_SIZE = 16

img_size = (224, 224) if 'Efficient_b4' not in model_path else (380, 380)
print(img_size)
test_set = HAMDataset(data_path, 'test', img_size)
test_loader = DataLoader(test_set,batch_size=BATCH_SIZE, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device = ',device)

## Model
NUM_CLASS = 7
model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=NUM_CLASS).to(device)
# model = models.resnet101(pretrained=True)
# model.fc = nn.Sequential(
#     # nn.Dropout(0.8),
#     nn.Linear(2048, NUM_CLASS)
# )
model.load_state_dict(torch.load(model_path))

model = model.to(device)
model.eval()

test_pred = torch.tensor([])
test_names = []

with torch.no_grad():
    for batch in tqdm(test_loader):
        imgs, names = batch
        imgs = imgs.to(device)
        logits = model(imgs)
        logits = torch.sigmoid(logits)

        if len(test_pred) == 0:
            test_pred = logits.cpu()
        else:
            test_pred = torch.cat((test_pred, logits.cpu()), 0)
        test_names += list(names)

test_pred = test_pred.numpy()

with open(output_path, 'w') as f:
    f.write('image,MEL,NV,BCC,AKIEC,BKL,DF,VASC\n')
    for i, y in enumerate(test_pred):
        f.write('{},{},{},{},{},{},{},{}\n'.format(test_names[i][:-4], y[0],y[1],y[2],y[3],y[4],y[5],y[6]))