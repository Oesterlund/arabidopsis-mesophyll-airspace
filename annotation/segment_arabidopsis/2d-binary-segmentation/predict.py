# pylint: disable=missing-module-docstring, missing-function-docstring
import argparse
import os

import numpy as np
import torch

from skimage.io import imsave
#import napari

from net_wrapper import NetWrapper
from data_module import DataModule

from parameters import get_params # pylint: disable=import-error, wrong-import-order



def main():
    # pylint: disable=too-many-locals
    parser = argparse.ArgumentParser()
    parser.add_argument('model_checkpoint', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('version', type=int, choices=(0,))
    parser.add_argument('--device', type=str, choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()

    params = get_params(args.version, predict=True)
    data_module = DataModule(
        **params,
        num_workers=16,
        extra_data_loader_kwargs={'pin_memory' : True, 'pin_memory_device' : args.device},
    )
    data_module.prepare_data()
    data_module.setup()
    model = NetWrapper.load_from_checkpoint(args.model_checkpoint).to(args.device)
    model.eval()
    loaders = {}
    for ds_name in ('train', 'validation', 'test', 'predict'):
        try:
            loaders[ds_name] = data_module._get_dataloader(ds_name)
            os.makedirs(os.path.join(args.outdir, ds_name, 'images'), exist_ok=True)
            os.makedirs(os.path.join(args.outdir, ds_name, 'preds'), exist_ok=True)
        except KeyError:
            pass

    with torch.no_grad():
        for ds_name, loader in loaders.items():
            for batch in loader:
                preds, images, paths = model.predict_step(batch, return_filename=True, device=args.device)
                for i, path in enumerate(paths):
                    outpath = os.path.join(args.outdir, ds_name, 'preds', os.path.basename(path))
                    pred = (preds[i].squeeze().cpu().numpy().T > 0.5).astype('uint8')
                    imsave(outpath, 255*pred, check_contrast=False)
                    outpath = os.path.join(args.outdir, ds_name, 'images', os.path.basename(path))
                    im = np.round(255*images[i].squeeze().cpu().numpy().T).astype('uint8')
                    imsave(outpath, im, check_contrast=False)
                    
if __name__ == '__main__':
    main()
