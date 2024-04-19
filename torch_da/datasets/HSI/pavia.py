import os
from hsicls.datasets.utils import open_file

rgb_bands = (43, 21, 11)
palette = None
label_values  = ['Unlabeled', 'Tree', 'Asphalt', 'Brick', 'Bitumen', 'Shadow', 'Meadows', 'Bare Soil']

def paviau_loader(folder):
    folder = os.path.join(folder, 'Pavia')
    img = open_file(os.path.join(folder, 'PaviaU.mat'))['ori_data']
    gt = open_file(os.path.join(folder, 'PaviaU_7gt.mat'))['map']
    return img, gt, label_values, rgb_bands, palette

def paviac_loader(folder):
    folder = os.path.join(folder, 'Pavia')
    img = open_file(os.path.join(folder, 'PaviaC.mat'))['ori_data']
    gt = open_file(os.path.join(folder, 'PaviaC_7gt.mat'))['map']
    return img, gt, label_values, rgb_bands, palette
