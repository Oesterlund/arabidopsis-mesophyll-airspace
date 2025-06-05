'''See class DataModule'''
import os
import csv
import random
import gzip

import numpy as np

import torch
import torch.nn.functional as F
import torchio as tio

from base_data_module import BaseDataModule

__all__ = [
    'DataModule'
]

        
class DataModule(BaseDataModule):
    '''
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 data_dir,
                 data_info_path,
                 batch_size,
                 train_ratio=0.9,
                 val_ratio=0.1,
                 image_types=frozenset(('png', 'tiff')),
                 **kwargs
                 ):
        '''
        Parameters
        ----------
        data_dir : str
          The root data directory. It is expected to contain *nii.gz files

        data_info_path : str
          EITHER
           an existing csv file containing *at least* the following named columns
             dataset,path
           all other columns are ignored.
           The dataset column must contain values from {train, validation, test, predict}.
          OR
           a path to store data info in.
          If the file exists it will be used, if it does not exist it will be generated.

        batch_size : int
          Batch size
        '''
        # pylint: disable=too-many-arguments
        super().__init__(
            data_info_path,
            batch_size,
            **kwargs
        )
        self.data_dir = data_dir
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.image_types = image_types

    def create_subject(self, row):
        subject_kwargs = {
            'image' : tio.ScalarImage(row.image),
            'label' : tio.LabelMap(row.labels),
        }
        return tio.Subject(**subject_kwargs)

    def prepare_data(self):
        '''Do the following
        1. Create data info file
        '''
        if not os.path.exists(self.data_info_path):
            self._create_data_info()

    def _create_data_info(self):
        header = ['dataset', 'image', 'labels']
        image_paths, labels_paths = [], []
        label_dir = os.path.join(self.data_dir, 'labels/')
        image_dir = os.path.join(self.data_dir, 'images/')
        for entry in os.scandir(label_dir):
            #if any((entry.name.endswith(image_type) for image_type in self.image_types)):
            image_path = os.path.join(image_dir, entry.name)
            labels_path = os.path.join(label_dir, entry.name)
            image_paths.append(image_path)
            labels_paths.append(labels_path)
            #else:
            #    print(f'Skipping {entry.name}')
        n_images = len(image_paths)
        n_train = round(self.train_ratio * n_images)
        n_val = round(self.val_ratio * n_images)
        n_test = n_images - n_val - n_train
        datasets = ['train']*n_train + ['validation']*n_val + ['test']*n_test
        random.shuffle(datasets)

        with open(self.data_info_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(header)
            writer.writerows(
                zip(datasets, image_paths, labels_paths)
            )
