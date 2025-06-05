# pylint: disable=missing-module-docstring, missing-function-docstring
import argparse
import os
from datetime import datetime

import numpy as np

import pytorch_lightning as pl
from monai.networks.nets import UNet
from monai import losses

import torch


from net_wrapper import NetWrapper
from data_module import DataModule

from parameters import get_params # pylint: disable=import-error, wrong-import-order

#import napari

def main():
    description = '''
    Train a UNet to segment cells in phase contrast CT images of Arabidopsis leaves
    '''
    version_help = '''See parameters.py'''

    parser = argparse.ArgumentParser(description = description)
    parser.add_argument('version', type=int, choices=(0,1,2), help=version_help)
    parser.add_argument('--model-checkpoint', type=str, default=None)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--only-data-setup', action='store_true')
    parser.add_argument('--precision', choices=('medium', 'high', 'highest'), default='medium')
    args = parser.parse_args()

    torch.set_float32_matmul_precision(args.precision)
    
    params = get_params(args.version)
    data_module = DataModule(
        **params,
        num_workers=16,
        extra_data_loader_kwargs={'pin_memory' : True},
    )
    data_module.prepare_data()
    data_module.setup()
    if args.only_data_setup:
        return 0
    
    loss_function = losses.DiceLoss(sigmoid=True)
    if args.model_checkpoint is None:
        net = UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=[32,32,64,64,128,128,256],
            strides=(1,2,1,2,1,2),
            num_res_units=3,
        )
        model = NetWrapper(net, loss_function, learning_rate=args.learning_rate)
    else:
        model = NetWrapper.load_from_checkpoint(args.model_checkpoint)
        model.loss = loss_function
        
    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        filename='{epoch}-{step}-{train_loss:.2f}-{val_loss:.2f}',
        mode='min',
        save_top_k=10,
        save_last=True,
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=-1,
        log_every_n_steps=1,
        callbacks=[model_checkpoint],
        max_epochs=args.max_epochs,
        default_root_dir=os.path.join(os.getcwd(), 'logs', f'{args.version:02d}')
    )

    start = datetime.now()
    print('Training started at', start)
    if args.model_checkpoint is None:
        trainer.fit(model=model, datamodule=data_module)
    else:
        trainer.fit(ckpt_path=args.model_checkpoint, model=model, datamodule=data_module)
    print('Training duration:', datetime.now() - start)

if __name__ == '__main__':
    main()
