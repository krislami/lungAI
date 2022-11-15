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

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn import model_selection
import albumentations as A
import timm
from omegaconf import OmegaConf

import glob
from sklearn.metrics import roc_auc_score
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
             'epoch': 100,
             'image_size': 512,
             'model_name': 'tf_efficientnet_b3_ns',
             'lr': 0.001,
             'fold': 0,
             'ckpt_pth': None,
             'data_dir': None,
             'add_data_dir': None,
             'add_data_dir2': None,
             'output_dir': './',
             'seed': 2021,
             'trainer': {}}
conf_base = OmegaConf.create(conf_dict)


####################
# Dataset
####################
class NagasakiClusterDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.labels_list = ['Papillary', 'OtherCarcinomas', 'NoCarcinomaCells', 'InvasiveMucinous', 'Acinar', 'Micropapillary', 'Lepidic', 'Solid']
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
        dist_label = torch.from_numpy(self.data.loc[idx, self.labels_list].values.astype(np.float)).float()
        dist_label = dist_label/dist_label.sum()
        return image, label, dist_label
           
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
    def setup(self, stage=None):
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
            
            # add dist label
            dist_df = pd.read_csv('dataset/12WSI_cluster1_dist.csv')
            if 'cluster2' in self.conf.data_dir:
                dist_df = pd.read_csv('dataset/12WSI_cluster2_dist.csv')
            labels_list = ['Papillary', 'OtherCarcinomas', 'NoCarcinomaCells', 'InvasiveMucinous', 'Acinar', 'Micropapillary', 'Lepidic', 'Solid']
            df[labels_list] = 0
            for idx, item in df.iterrows():
                df.loc[idx, labels_list] = dist_df[dist_df['image'] == os.path.basename(item['image_path'])].values[0][1:]

            # cv split
            skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2021)
            for n, (train_index, val_index) in enumerate(skf.split(df, df["class_id"], df["wsi"])):
                df.loc[val_index, 'fold'] = int(n)
            df['fold'] = df['fold'].astype(int)           
            
            train_df = df[df["fold"] != self.conf.fold]
            valid_df = df[df["fold"] == self.conf.fold]
            
            ## add 79WSI data
            add_df = pd.read_csv('dataset/79WSI_cluster1.csv')
            if 'cluster2' in self.conf.data_dir:
                add_df = pd.read_csv('dataset/79WSI_cluster2.csv')
            
            add_image_list = sorted(glob.glob(os.path.join(self.conf.add_data_dir, '*.png')))
            add_image_list = [im for im in add_image_list if '_'.join(im.split('/')[-1].split('_')[:2]) in add_df['wsi'].values]
            add_wsi_list = ['_'.join(im.split('/')[-1].split('_')[:2]) for im in add_image_list]
            add_label_list = [add_df[add_df['wsi']==wsi]['class'].values[0] for wsi in add_wsi_list]
            
            add_train_df= pd.DataFrame(list(zip(add_image_list, add_label_list, add_wsi_list)), columns = ['image_path', 'class', 'wsi'])
            add_train_df['class_id'] = add_train_df['class'].map(label_dict)
            
            print(f'added {len(add_train_df)} images !!')
            
            # add dist label
            dist_df = pd.read_csv('dataset/79WSI_cluster1_dist.csv')
            if 'cluster2' in self.conf.data_dir:
                dist_df = pd.read_csv('dataset/79WSI_cluster2_dist.csv')
            labels_list = ['Papillary', 'OtherCarcinomas', 'NoCarcinomaCells', 'InvasiveMucinous', 'Acinar', 'Micropapillary', 'Lepidic', 'Solid']
            add_train_df[labels_list] = 0
            for idx, item in add_train_df.iterrows():
                add_train_df.loc[idx, labels_list] = dist_df[dist_df['wsi'] == item['wsi']].values[0][1:]
                
            # add No Carcinoma Patch
            add_image_list = sorted(glob.glob(os.path.join(self.conf.add_data_dir2, '*.png')))
            add_wsi_list = ['_'.join(im.split('/')[-1].split('_')[:2]) for im in add_image_list]
            add_label_list = ['NoCarcinomaCells' for wsi in add_wsi_list]
            add_train_df2= pd.DataFrame(list(zip(add_image_list, add_label_list, add_wsi_list)), columns = ['image_path', 'class', 'wsi'])
            add_train_df2['class_id'] = add_train_df2['class'].map(label_dict)
            
            for idx, item in add_train_df2.iterrows():
                add_train_df2.loc[idx, labels_list] = np.array([0,0,1,0,0,0,0,0])
                
            print(f'added {len(add_train_df2)} images !!')
            
            
            
            train_df = pd.concat([train_df, add_train_df, add_train_df2])
            print(f'total {len(train_df)} images !!')
            
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
                        #A.Resize(size, size),
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
            
        elif stage == 'test':
            pass
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.conf.batch_size, num_workers=4, shuffle=False, pin_memory=True, drop_last=True)

    def test_dataloader(self):
        pass
        
####################
# Lightning Module
####################
#　ここまで
class LitSystem(pl.LightningModule):
    def __init__(self, conf):
        super().__init__()
        self.save_hyperparameters(conf)
        self.model = timm.create_model(model_name=self.hparams.model_name, num_classes=8, pretrained=True, in_chans=3)
        if self.hparams.ckpt_pth is not None:
            self.model = load_pytorch_model(self.hparams.model_ckpt, self.model)
        self.criteria = torch.nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy()

    def forward(self, x):
        # use forward for inference/predictions
        return self.model(x)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.epoch)
        
        return [optimizer], [{'scheduler': scheduler, 'interval': 'epoch'}]

    def training_step(self, batch, batch_idx):
        x, y, dist_label = batch
        
        if torch.rand(1)[0] < 0.5: #self.current_epoch < self.hparams.epoch*1.8:
            # mixup
            alpha = 0.5
            lam = np.random.beta(alpha, alpha)
            batch_size = x.size()[0]
            index = torch.randperm(batch_size)
            x = lam * x + (1 - lam) * x[index, :]

            y_hat = self.model(x)
            loss = lam * self.criteria(y_hat, dist_label) + (1 - lam) * self.criteria(y_hat, dist_label[index])
        else:
            y_hat = self.model(x)
            loss = self.criteria(y_hat, dist_label)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, dist_label = batch
        y_hat = self.model(x)
        loss = self.criteria(y_hat, dist_label)
        
        return {
            "val_loss": loss,
            "y": y,
            "y_hat": y_hat
            }
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        y = torch.cat([x["y"] for x in outputs]).cpu()
        y_hat = torch.cat([x["y_hat"] for x in outputs]).cpu()

        preds = np.argmax(y_hat, axis=1)
        
        val_acc = self.accuracy(y, preds)

        self.log('avg_val_loss', avg_val_loss)
        self.log('val_acc', val_acc)
        
        
####################
# Train
####################  
def main():
    conf_cli = OmegaConf.from_cli()
    conf = OmegaConf.merge(conf_base, conf_cli)
    print(OmegaConf.to_yaml(conf))
    seed_everything(2021)

    tb_logger = loggers.TensorBoardLogger(save_dir=os.path.join(conf.output_dir, 'tb_log/'))
    csv_logger = loggers.CSVLogger(save_dir=os.path.join(conf.output_dir, 'csv_log/'))

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(conf.output_dir, 'ckpt/'), monitor='val_acc', 
                                          save_last=True, save_top_k=1, mode='max', 
                                          save_weights_only=True, filename='{epoch}-{val_acc:.5f}')

    data_module = NagasakiClusterDataModule(conf)

    lit_model = LitSystem(conf)

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=conf.epoch,
        gpus=-1,
        amp_backend='native',
        amp_level='O2',
        precision=16,
        num_sanity_val_steps=10,
        val_check_interval=1.0,
        **conf.trainer
            )

    trainer.fit(lit_model, data_module)

if __name__ == "__main__":
    main()