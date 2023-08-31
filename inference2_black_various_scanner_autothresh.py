import argparse
import glob
import os
import sys

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import timm
import torch
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str)
parser.add_argument('model_dir', type=str)
parser.add_argument('output_dir', type=str)
parser.add_argument('--mpp', type=float)
parser.add_argument('--bgthresh', type=float, default=0.25)
args = parser.parse_args()

print(args)

data_dir = args.data_dir #sys.argv[1] #'/kqi/git/data/Test/'
model_dir= args.model_dir #sys.argv[2] #'/kqi/parent/22022651/'
output_dir= args.output_dir #sys.argv[3]
arg_mpp= args.mpp
bgthresh= args.bgthresh

label_dict = {'Papillary': 0,
             'OtherCarcinomas': 1,
             'NoCarcinomaCells': 2,
             'InvasiveMucinous': 3,
             'Acinar': 4,
             'Micropapillary': 5,
             'Lepidic': 6,
             'Solid':7}

color_dict = {0: [255, 255, 0],
             1: [165, 165, 165],
             2: [255, 255, 255],
             3: [112, 48, 160],
             4: [0, 176, 80],
             5: [255, 192, 0],
             6: [0, 176, 240],
             7: [255, 0, 0]}

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
Image.MAX_IMAGE_PIXELS = None
target_list = sorted(glob.glob(os.path.join(data_dir,'*.*')))

model_path = []
for i in range(5):
    target_model = glob.glob(os.path.join(model_dir,  f'fold{i}/ckpt/*epoch*.ckpt'))
    model_path += target_model

models = []
for ckpt in model_path:
    m = timm.create_model(model_name='tf_efficientnet_b3_ns', num_classes=8, pretrained=False, in_chans=3)
    m = load_pytorch_model(ckpt, m, ignore_suffix='model')
    m.cuda()
    m.eval()
    models.append(m)

transform = A.Compose([
            A.Resize(height=512, width=512, always_apply=False, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0)
            ])


result_df = pd.DataFrame(columns=['numAllPatch','numPapillary', 'numOtherCarcinomas', 'numNoCarcinomaCells', 'numInvasiveMucinous',
                              'numAcinar', 'numMicropapillary', 'numLepidic', 'numSolid'])
result_black_df = pd.DataFrame(columns=['numAllPatch','numPapillary', 'numOtherCarcinomas', 'numNoCarcinomaCells', 'numInvasiveMucinous',
                              'numAcinar', 'numMicropapillary', 'numLepidic', 'numSolid'])
result_file_list = []
for file_name in target_list:
    print(file_name)
    wsi = openslide.OpenSlide(file_name)
    if openslide.PROPERTY_NAME_MPP_X not in list(dict(wsi.properties).keys()):
        print(f'ERROR!! NO MPP INFORMATION IN THE WSI: {file_name}')
        if arg_mpp is not None:
            mpp=arg_mpp
        else:
            continue
    else:
        print(f'NOW PROCESSING FOR {file_name}')
        print(f'MPP: {float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])}')
        mpp = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X])
    tile_size = int(1000//mpp)
    dimension = wsi.level_dimensions[0]

    tissue_map = wsi.get_thumbnail((int(dimension[0]//100), int(dimension[1]//100))).crop((0, 0, int((tile_size * dimension[0]//tile_size)/100), int((tile_size * dimension[1]//tile_size)/100))).convert('L')
    tissue_map = cv2.adaptiveThreshold(np.array(tissue_map),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    tissue_map = cv2.morphologyEx(tissue_map, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    tissue_map = cv2.resize(tissue_map, (int(64 * (dimension[0]//tile_size)) ,int(64 * (dimension[1]//tile_size))))/255

    prob_map = np.zeros((int(64 * (dimension[1]//tile_size)) ,int(64 * (dimension[0]//tile_size)),3), dtype=np.uint8)
    prob_map_black = np.zeros((int(64 * (dimension[1]//tile_size)) ,int(64 * (dimension[0]//tile_size)),3), dtype=np.uint8)
    num_result = [0,0,0,0,0,0,0,0,0]
    num_result_black = [0,0,0,0,0,0,0,0,0]
    with torch.no_grad():
        for iw in range(int(dimension[0]//tile_size)):
            for ih in range(int(dimension[1]//tile_size)):
                wsi_region = wsi.read_region((iw*tile_size, ih*tile_size), 0, (tile_size,tile_size))

                wsi_region_pt = torch.from_numpy(transform(image=np.array(wsi_region)[:,:,:3])["image"].transpose(2, 0, 1))
                prob = np.zeros((5,8))
                for idx, m in enumerate(models):
                    prob[idx] = m(wsi_region_pt.cuda().unsqueeze(dim=0))[0].cpu().softmax(dim=0)
                prob = prob.mean(axis=0).argmax()
                prob_map[ih*64:(1+ih)*64, iw*64:(1+iw)*64, :] = color_dict[prob]
                prob_map_black[ih*64:(1+ih)*64, iw*64:(1+iw)*64, :] = color_dict[prob]
                num_result[0] += 1
                num_result[prob+1] += 1

                #if np.array(wsi_region)[:,:,1].mean()>bgthresh or np.array(wsi_region)[:,:,3].min()==0.0:
                #print(tissue_map[ih*496:(1+ih)*496, iw*496:(1+iw)*496].mean())
                if tissue_map[ih*64:(1+ih)*64, iw*64:(1+iw)*64].mean()<bgthresh or np.array(wsi_region)[:,:,3].min()==0.0:
                    num_result_black[0] -= 1
                    num_result_black[prob+1] -= 1
                    prob_map_black[ih*64:(1+ih)*64, iw*64:(1+iw)*64, :] = [0, 0, 0]
    
    del tissue_map

    result_df = result_df.append(pd.Series(num_result, index=result_df.columns), ignore_index=True)
    result_black_df = result_black_df.append(pd.Series(np.array(num_result) + np.array(num_result_black), index=result_black_df.columns), ignore_index=True)
    result_file_list.append(file_name)
    print('INFERENCE END')
    fg = wsi.get_thumbnail((int(dimension[0]//16), int(dimension[1]//16))).crop((0, 0, int((tile_size * dimension[0]//tile_size)/16), int((tile_size * dimension[1]//tile_size)/16))).convert('RGB')
    bg = Image.fromarray(cv2.resize(prob_map, fg.size)).convert('RGB')
    bg_black = Image.fromarray(cv2.resize(prob_map_black, fg.size)).convert('RGB')
    
    fg.save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_thumb.png'))
    bg.save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_prob.png'))
    bg_black.save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_prob_noTissue.png'))
    
    Image.blend(fg, bg, 0.5).save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_blend.png'))
    Image.blend(fg, bg_black, 0.5).save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_blend_noTissue.png'))

result_file_list = [os.path.basename(t) for t in result_file_list]
result_df['wsi'] = result_file_list
result_black_df['wsi'] = result_file_list
result_df[['wsi', 'numAllPatch','numPapillary', 'numOtherCarcinomas', 'numNoCarcinomaCells', 'numInvasiveMucinous',
    'numAcinar', 'numMicropapillary', 'numLepidic', 'numSolid']].to_csv(os.path.join(output_dir, 'result.csv'), index=False)
result_black_df[['wsi', 'numAllPatch','numPapillary', 'numOtherCarcinomas', 'numNoCarcinomaCells', 'numInvasiveMucinous',
    'numAcinar', 'numMicropapillary', 'numLepidic', 'numSolid']].to_csv(os.path.join(output_dir, 'result-noTissue.csv'), index=False)
