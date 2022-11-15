import openslide
import glob
import cv2
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

data_dir = sys.argv[1]
output_dir = sys.argv[2]

target_list = [os.path.basename(item).split('.')[0]for item in glob.glob(os.path.join(data_dir,'*.xml'))]

for target in target_list:
    print(f'Processing: {target}')
    annotation = openslide.OpenSlide(os.path.join(data_dir,target+'.tiff'))
    wsi = openslide.OpenSlide(os.path.join(data_dir,target+'.svs'))
    
    dimension = wsi.level_dimensions[1]
    
    for iw in range(dimension[0]//496):
        for ih in range(dimension[1]//496):
            ann_region = annotation.read_region((iw*496*4, ih*496*4), 2, (496,496))
            if np.array(ann_region)[:,:,0].max()==0:
                continue
            if np.array(ann_region)[:,:,0].sum() < 496*496*0.25:
                continue
            wsi_region = wsi.read_region((iw*496*4, ih*496*4), 1, (496,496))

            patch_name = f'{target}_{ih*496}_{iw*496}_{ih*496+496}_{iw*496+496}.png'

            wsi_region.save(os.path.join(output_dir, patch_name))