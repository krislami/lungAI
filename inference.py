import openslide
import glob
import cv2
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import timm
import torch
import albumentations as A
import sys
import pandas as pd

data_dir = sys.argv[1] #ex: '/kqi/git/data/Test/'
model_dir= sys.argv[2] #ex: '/kqi/parent/22022651/'
output_dir= sys.argv[3]

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

# Utils
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

target_list = sorted(glob.glob(os.path.join(data_dir,'*.svs')))


# model load
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

# Run Inference
for file_name in target_list:
    wsi = openslide.OpenSlide(file_name)
    dimension = wsi.level_dimensions[1]
    prob_map = np.zeros((int(496 * (dimension[1]//496)) ,int(496 * (dimension[0]//496)),3), dtype=np.uint8)
    prob_map_black = np.zeros((int(496 * (dimension[1]//496)) ,int(496 * (dimension[0]//496)),3), dtype=np.uint8)
    num_result = [0,0,0,0,0,0,0,0,0]
    num_result_black = [0,0,0,0,0,0,0,0,0]
    with torch.no_grad():
        for iw in range(dimension[0]//496):
            for ih in range(dimension[1]//496):
                wsi_region = wsi.read_region((iw*496*4, ih*496*4), 1, (496,496))

                wsi_region_pt = torch.from_numpy(transform(image=np.array(wsi_region)[:,:,:3])["image"].transpose(2, 0, 1))
                prob = np.zeros((5,8))
                for idx, m in enumerate(models):
                    prob[idx] = m(wsi_region_pt.cuda().unsqueeze(dim=0))[0].cpu().softmax(dim=0)
                prob = prob.mean(axis=0).argmax()
                prob_map[ih*496:(1+ih)*496, iw*496:(1+iw)*496, :] = color_dict[prob]
                prob_map_black[ih*496:(1+ih)*496, iw*496:(1+iw)*496, :] = color_dict[prob]
                num_result[0] += 1
                num_result[prob+1] += 1

                if np.array(wsi_region)[:,:,1].mean()>235:
                    num_result_black[0] -= 1
                    num_result_black[prob+1] -= 1
                    prob_map_black[ih*496:(1+ih)*496, iw*496:(1+iw)*496, :] = [0, 0, 0]
                  
    result_df = result_df.append(pd.Series(num_result, index=result_df.columns), ignore_index=True)
    result_black_df = result_black_df.append(pd.Series(np.array(num_result) + np.array(num_result_black), index=result_black_df.columns), ignore_index=True)

    fg = wsi.read_region((0, 0), 2, (int(496 * (dimension[0]//496)/4) ,int(496 * (dimension[1]//496)/4))).convert('RGB')
    bg = Image.fromarray(cv2.resize(prob_map, (int(496 * (dimension[0]//496)/4) ,int(496 * (dimension[1]//496)/4)))).convert('RGB')
    bg_black = Image.fromarray(cv2.resize(prob_map_black, (int(496 * (dimension[0]//496)/4) ,int(496 * (dimension[1]//496)/4)))).convert('RGB')

    
    fg.save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_thumb.png'))
    bg.save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_prob.png'))
    bg_black.save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_prob_noTissue.png'))
    
    Image.blend(fg, bg, 0.5).save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_blend.png'))
    Image.blend(fg, bg_black, 0.5).save(os.path.join(output_dir, os.path.basename(file_name).split('.')[0]+'_blend_noTissue.png'))

# save inference results
target_list = [os.path.basename(t) for t in target_list]
result_df['wsi'] = target_list
result_black_df['wsi'] = target_list
result_df[['wsi', 'numAllPatch','numPapillary', 'numOtherCarcinomas', 'numNoCarcinomaCells', 'numInvasiveMucinous',
    'numAcinar', 'numMicropapillary', 'numLepidic', 'numSolid']].to_csv('/kqi/output/result.csv', index=False)
result_df[['wsi', 'numAllPatch','numPapillary', 'numOtherCarcinomas', 'numNoCarcinomaCells', 'numInvasiveMucinous',
    'numAcinar', 'numMicropapillary', 'numLepidic', 'numSolid']].to_csv(os.path.join(output_dir, 'result.csv'), index=False)
result_black_df[['wsi', 'numAllPatch','numPapillary', 'numOtherCarcinomas', 'numNoCarcinomaCells', 'numInvasiveMucinous',
    'numAcinar', 'numMicropapillary', 'numLepidic', 'numSolid']].to_csv('/kqi/output/result-noTissue.csv', index=False)
result_black_df[['wsi', 'numAllPatch','numPapillary', 'numOtherCarcinomas', 'numNoCarcinomaCells', 'numInvasiveMucinous',
    'numAcinar', 'numMicropapillary', 'numLepidic', 'numSolid']].to_csv(os.path.join(output_dir, 'result-noTissue.csv'), index=False)