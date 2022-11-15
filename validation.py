####################
# Import Libraries
####################
import os
import sys
import cv2
import numpy as np
import pandas as pd

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning import loggers
from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import torch
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn import model_selection
import albumentations as A
import timm
from omegaconf import OmegaConf

import glob
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
####################
# Utils
####################
def load_pytorch_model(ckpt_name, model, ignore_suffix='model'):
    state_dict = torch.load(ckpt_name, map_location='cpu')["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k
        if name.startswith(str(ignore_suffix)+"."):
            name = name.replace(str(ignore_suffix)+".", "", 1)  # remove `model.`
        new_state_dict[name] = v
    res = model.load_state_dict(new_state_dict, strict=False)
    print(res)
    return model

####################
# Config
####################

conf_dict = {'batch_size': 64, 
             'image_size': 512,
             'model_name': 'tf_efficientnet_b3_ns',
             'fold': 0,
             'model_dir': None,
             'data_dir': None,
             'output_dir': './',
            'seed': 2021}
conf_base = OmegaConf.create(conf_dict)

####################
# Dataset
####################
class NagasakiClusterDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        cv2.setNumThreads(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, "image_path"]
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)
        image = torch.from_numpy(image["image"].transpose(2, 0, 1))
        label = self.data.loc[idx, "class_id"]
        return image, label
           
####################
# Data Module
####################

class NagasakiClusterDataModule(pl.LightningDataModule):

    def __init__(self, conf):
        super().__init__()
        self.conf = conf       

    # OPTIONAL, called only on 1 GPU/machine(for download or tokenize)
    def prepare_data(self):
        pass

    # OPTIONAL, called for every GPU/machine
    def setup(self, stage=None, fold=None):
        if stage == 'fit':
            image_list = sorted(glob.glob(os.path.join(self.conf.data_dir, '*/*.jpg')))
            label_list = [im.split('/')[-2] for im in image_list]
            wsi_list = [im.split('/')[-1].split('_')[0] for im in image_list]
            label_dict = {'Papillary': 0,
                         'OtherCarcinomas': 1,
                         'NoCarcinomaCells': 2,
                         'InvasiveMucinous': 3,
                         'Acinar': 4,
                         'Micropapillary': 5,
                         'Lepidic': 6,
                         'Solid':7}
            df= pd.DataFrame(list(zip(image_list, label_list, wsi_list)), columns = ['image_path' , 'class', 'wsi'])
            df['class_id'] = df['class'].map(label_dict)
            
             # cv split
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2021)
            for n, (train_index, val_index) in enumerate(skf.split(df, df["class_id"], df["wsi"])):
                df.loc[val_index, 'fold'] = int(n)
            df['fold'] = df['fold'].astype(int)           
            
            train_df = df[df["fold"] != fold]
            valid_df = df[df["fold"] == fold]
            
            train_transform = A.Compose([
                        A.Resize(height=self.conf.image_size, width=self.conf.image_size, p=1), 
                        A.Flip(p=0.5),
                        A.ShiftScaleRotate(p=0.5),
                        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.7),
                        A.RandomBrightnessContrast(brightness_limit=(-0.3,0.3), contrast_limit=(-0.3, 0.3), p=0.8),
                        A.CLAHE(clip_limit=(1,4), p=0.5),
                        A.OneOf([
                            A.OpticalDistortion(distort_limit=1.0),
                            A.GridDistortion(num_steps=5, distort_limit=1.),
                            A.ElasticTransform(alpha=3),
                        ], p=0.20),
                        A.OneOf([
                            A.GaussNoise(var_limit=[10, 50]),
                            A.GaussianBlur(),
                            A.MotionBlur(),
                            A.MedianBlur(),
                        ], p=0.20),
                        A.OneOf([
                            A.JpegCompression(quality_lower=95, quality_upper=100, p=0.50),
                            A.Downscale(scale_min=0.75, scale_max=0.95),
                        ], p=0.2),
                        A.IAAPiecewiseAffine(p=0.2),
                        A.IAASharpen(p=0.2),
                        A.Cutout(max_h_size=int(self.conf.image_size * 0.1), max_w_size=int(self.conf.image_size * 0.1), num_holes=5, p=0.5),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])

            valid_transform = A.Compose([
                        A.Resize(height=self.conf.image_size, width=self.conf.image_size, always_apply=False, p=1.0),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
                        ])
            
            self.train_dataset = NagasakiClusterDataset(train_df, transform=train_transform)
            self.valid_dataset = NagasakiClusterDataset(valid_df, transform=valid_transform)
            self.valid_df = valid_df
            
        elif stage == 'test':
            pass
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        pass
        
# ====================================================
# Inference function
# ====================================================
def inference(models, test_loader):
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    probs = []
    prob_labels = []
    with torch.no_grad():
        for i, (images) in tk0:
            images = images[0].cuda()
            avg_preds = []
            for model in models:
                y_preds = torch.softmax(model(images), dim=1)

                avg_preds.append(y_preds.to('cpu').numpy())

            avg_preds = np.mean(avg_preds, axis=0)
            #probs.append(np.max(avg_preds, axis=1))
            probs.append(avg_preds)
            prob_labels.append(np.argmax(avg_preds, axis=1))
        probs = np.concatenate(probs)
        prob_labels = np.concatenate(prob_labels)
    return probs, prob_labels
  
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)
    labels_list = ['Papillary', 'OtherCarcinomas', 'NoCarcinomaCells', 'InvasiveMucinous', 'Acinar', 'Micropapillary', 'Lepidic', 'Solid']

    # get model path
    model_path = []
    for i in range(5):
        target_model = glob.glob(os.path.join(conf.model_dir,  f'fold{i}/ckpt/*epoch*.ckpt'))
        model_path += target_model
        
    models = []
    for ckpt in model_path:
        m = timm.create_model(model_name=conf.model_name, num_classes=8, pretrained=True, in_chans=3)
        m = load_pytorch_model(ckpt, m, ignore_suffix='model')
        m.cuda()
        m.eval()
        models.append(m)

    
    # make oof
    oof_df = pd.DataFrame()
    for f, m in enumerate(models):
        data_module = NagasakiClusterDataModule(conf)
        data_module.setup(stage='fit', fold=f)
        valid_df = data_module.valid_df
        valid_dataset = data_module.valid_dataset
        valid_loader =  DataLoader(valid_dataset, batch_size=conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=False)
        
        predictions, prediction_labels = inference([m], valid_loader)
        print(predictions.shape)
        #valid_df['preds'] = predictions
        valid_df['preds_'+labels_list[0]] = predictions[:,0]
        valid_df['preds_'+labels_list[1]] = predictions[:,1]
        valid_df['preds_'+labels_list[2]] = predictions[:,2]
        valid_df['preds_'+labels_list[3]] = predictions[:,3]
        valid_df['preds_'+labels_list[4]] = predictions[:,4]
        valid_df['preds_'+labels_list[5]] = predictions[:,5]
        valid_df['preds_'+labels_list[6]] = predictions[:,6]
        valid_df['preds_'+labels_list[7]] = predictions[:,7]
        valid_df['prediction_labels'] = prediction_labels
        oof_df = pd.concat([oof_df, valid_df])


    oof_score = accuracy_score(oof_df['class_id'], oof_df['prediction_labels'])
    oof_df.to_csv(os.path.join(conf.output_dir,  f"oof-{oof_score}.csv"), index=False)
    conf_matrix = confusion_matrix(oof_df['class_id'], oof_df['prediction_labels'])
    
    print(conf_matrix)
        
    print(oof_score)
    print(oof_df.head())
    print(model_path)
    
    

if __name__ == "__main__":
    main()