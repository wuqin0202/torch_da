import os
from hsicls.datasets.utils import open_file

rgb_bands = (43, 21, 11)
palette = None
label_values  = ['Unlabeled', 'Dense Urban Fabric', 'Mineral Extraction Sites',
                 'Non-irrigated Arable Land', 'Fruit Trees', 'Olive Groves',
                 'Coniferous Forest', 'Dense Sderophyllous Vegetation',
                 'Sparce Sderophyllous Vegetation', 'Sparcely Vegetated Areas',
                 'Rocks and Sand', 'Water', 'Coastal Water']

def dioni_loader(folder):
    folder = os.path.join(folder, 'HyRANK')
    img = open_file(os.path.join(folder, 'Dioni.mat'))['ori_data']
    gt = open_file(os.path.join(folder, 'Dioni_gt_out68.mat'))['map']
    return img, gt, label_values, rgb_bands, palette

def loukia_loader(folder):
    folder = os.path.join(folder, 'HyRANK')
    img = open_file(os.path.join(folder, 'Loukia.mat'))['ori_data']
    gt = open_file(os.path.join(folder, 'Loukia_gt_out68.mat'))['map']
    return img, gt, label_values, rgb_bands, palette
