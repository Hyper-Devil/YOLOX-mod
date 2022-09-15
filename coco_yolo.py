import json
import glob
import os
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm
def make_folders(path='../out/'):
    # Create folders
 
    if os.path.exists(path):
        shutil.rmtree(path)  # delete output folder
    os.makedirs(path)  # make new output folder
    os.makedirs(path + os.sep + 'labels')  # make new labels folder
    os.makedirs(path + os.sep + 'images')  # make new labels folder
    return path
def convert_coco_json(json_dir='/cone_coco/annotations/'):
    dir = make_folders(path='out/')  # output directory
    jsons = glob.glob(json_dir + '*.json')
    coco80 = coco91_to_coco80_class()
 
    # Import json
    for json_file in sorted(jsons):
        fn = 'out/labels/%s/' % Path(json_file).stem.replace('instances_', '')  # folder name
        os.mkdir(fn)
        with open(json_file) as f:
            data = json.load(f)
 
        # Create image dict
        images = {'%g' % x['id']: x for x in data['images']}
 
        # Write labels file
        for x in tqdm(data['annotations'], desc='Annotations %s' % json_file):
            if x['iscrowd']:
                continue
 
            img = images['%g' % x['image_id']]
            h, w, f = img['height'], img['width'], img['file_name']
 
            # The Labelbox bounding box format is [top left x, top left y, width, height]
            box = np.array(x['bbox'], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
 
            if (box[2] > 0.) and (box[3] > 0.):  # if w > 0 and h > 0
                with open(fn + Path(f).stem + '.txt', 'a') as file:
                    file.write('%g %.6f %.6f %.6f %.6f\n' % (coco80[x['category_id'] - 1], *box))
def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    x = [0, 1, 2]
    return x
 
convert_coco_json()