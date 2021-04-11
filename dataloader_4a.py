from torch.utils.data import Dataset, DataLoader# For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageOps
import torch
import pandas as pd
from collections import namedtuple

n_class    = 27

# a label and all meta information
Label = namedtuple( 'Label' , [
    'name'        , 
    'level3Id'    , 
    'color'       , 
    ] )

labels = [
    #       name                     level3Id  color
    Label(  'road'                 ,    0  , (128, 64,128)  ),
    Label(  'drivable fallback'    ,    1  , ( 81,  0, 81)  ),
    Label(  'sidewalk'             ,    2  , (244, 35,232)  ),
    Label(  'non-drivable fallback',    3  , (152,251,152)  ),
    Label(  'person/animal'        ,    4  , (220, 20, 60)  ),
    Label(  'rider'                ,    5  , (255,  0,  0)  ),
    Label(  'motorcycle'           ,    6  , (  0,  0,230)  ),
    Label(  'bicycle'              ,   7  , (119, 11, 32)  ),
    Label(  'autorickshaw'         ,   8  , (255, 204, 54) ),
    Label(  'car'                  ,   9  , (  0,  0,142)  ),
    Label(  'truck'                ,  10 ,  (  0,  0, 70)  ),
    Label(  'bus'                  ,  11 ,  (  0, 60,100)  ),
    Label(  'vehicle fallback'     ,  12 ,  (136, 143, 153)),  
    Label(  'curb'                 ,   13 ,  (220, 190, 40)),
    Label(  'wall'                 ,  14 ,  (102,102,156)  ),
    Label(  'fence'                ,  15 ,  (190,153,153)  ),
    Label(  'guard rail'           ,  16 ,  (180,165,180)  ),
    Label(  'billboard'            ,   17 ,  (174, 64, 67) ),
    Label(  'traffic sign'         ,  18 ,  (220,220,  0)  ),
    Label(  'traffic light'        ,  19 ,  (250,170, 30)  ),
    Label(  'pole'                 ,  20 ,  (153,153,153)  ),
    Label(  'obs-str-bar-fallback' , 21 ,  (169, 187, 214) ),  
    Label(  'building'             ,  22 ,  ( 70, 70, 70)  ),
    Label(  'bridge/tunnel'        ,  23 ,  (150,100,100)  ),
    Label(  'vegetation'           ,  24 ,  (107,142, 35)  ),
    Label(  'sky'                  ,  25 ,  ( 70,130,180)  ),
    Label(  'unlabeled'            ,  26 ,  (  0,  0,  0)  ),
]   

class IddDataset_4a(Dataset):

    def __init__(self, csv_file, n_class=n_class, transforms_=None, aug = False):
        self.data      = pd.read_csv(csv_file)
        self.n_class   = n_class
        self.mode = csv_file
        
        # Add any transformations here
        
        # The following transformation normalizes each channel using the mean and std provided
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),]) 
        self.aug = aug
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        new_size = (256,128)
        img_name = self.data.iloc[idx, 0]
        img = Image.open(img_name).convert('RGB')
        #im1 = im.crop((left, top, right, bottom)) 
        #1080 * 1920
        label_name = self.data.iloc[idx, 1]
        label = Image.open(label_name)

        if self.aug:
            img = img.rotate(5)
            img = img.crop((img.size[0] // 2 - 800, img.size[1] // 2 - 450, img.size[0] // 2 + 800,  img.size[1] // 2 + 450))
            label = label.rotate(5)
            label = label.crop((label.size[0] // 2 - 800, label.size[1] // 2 - 450, label.size[0] // 2 + 800,  label.size[1] // 2 + 450))
            #img = ImageOps.mirror(img)
            #label = ImageOps.mirror(label)
         
        img = img.resize(new_size, resample = Image.NEAREST)
        label = label.resize(new_size, resample = Image.NEAREST)
        
        img = np.asarray(img) / 255. # scaling [0-255] values to [0-1]
        label = np.asarray(label)
        
        img = self.transforms(img).float() # Normalization
        label = torch.from_numpy(label.copy()).long() # convert to tensor
        
        
            
        # create one-hot encoding
        h, w = label.shape
        target = torch.zeros(self.n_class, h, w)
        for c in range(self.n_class):
            target[c][label == c] = 1
        
        return img, target, label