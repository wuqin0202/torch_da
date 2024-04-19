import os
from hsicls.datasets.utils import open_file

rgb_bands = (43, 21, 11)
palette = None
label_values  = ['Unlabeled', 'Grass Healthy', 'Grass Stressed', 'Trees', 'Water', 'Residential Buildings', 'Non-residential Buildings', 'Roads']

def houston13_loader(folder):
    folder = os.path.join(folder, 'Houston')
    img = open_file(os.path.join(folder, 'Houston13.mat'))['ori_data']
    gt = open_file(os.path.join(folder, 'Houston13_7gt.mat'))['map']
    return img, gt, label_values, rgb_bands, palette

def houston18_loader(folder):
    folder = os.path.join(folder, 'Houston')
    img = open_file(os.path.join(folder, 'Houston18.mat'))['ori_data']
    gt = open_file(os.path.join(folder, 'Houston18_7gt.mat'))['map']
    return img, gt, label_values, rgb_bands, palette
