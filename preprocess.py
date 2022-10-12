import torch
import os
import glob
import pandas

root_pth = '/dataset/'

filenames = os.listdir(root_pth)
filenames = [filename for filename in filenames if '.' not in filename]

label2idx = {}
num = 0
for filename in filenames:
    label2idx[filename] = num
    num += 1

print(label2idx)

train_dict = {}
valid_dict = {}

count = 0 
for filename in filenames:
    images = os.listdir(os.path.join(root_pth, filename))
    for image in images:
        if count % 10 != 0:
            train_dict[os.path.join(root_pth,filename,image)] = label2idx[filename]
        else:
            valid_dict[os.path.join(root_pth,filename,image)] = label2idx[filename]
        count += 1

with open('train.csv','w') as f:
    f.write('img_path,label\n')
    for y in train_dict:
        f.write('{},{}\n'.format(y, train_dict[y]))

with open('valid.csv','w') as f:
    f.write('img_path,label\n')
    for y in valid_dict:
        f.write('{},{}\n'.format(y, valid_dict[y]))
        