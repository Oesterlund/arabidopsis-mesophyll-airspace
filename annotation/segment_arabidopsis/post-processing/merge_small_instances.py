import argparse
import os
import nibabel
import numpy as np
from skimage.segmentation import expand_labels
import matplotlib.pyplot as plt
import napari

def main():
    description = '''Merge small instances into non-small instances by expanding non-small instances
    into small instances'''
    parser = argparse.ArgumentParser(description)
    parser.add_argument('inpath', type=str, help='Path to instance segmentation')
    parser.add_argument('outpath', type=str, help='Path to store post-processed segmentation')
    parser.add_argument('--small-threshold', type=int, required=True,
                        help='Everything less than the threshold is considered small and is merged')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    
    outdir = os.path.dirname(args.outpath)
    if outdir != '':
        os.makedirs(outdir, exist_ok=True)
    
    seg = nibabel.load(args.inpath)
    affine = seg.affine
    seg = np.asanyarray(seg.dataobj)
    cells = seg > 0
    n_labels = np.max(seg)
    bins = np.arange(1,n_labels+2)
    sizes, _ = np.histogram(seg, bins=bins, range=(1, n_labels))
    bins = bins[:-1]

    small_instances = bins[sizes < args.small_threshold]
    seg[np.isin(seg, small_instances)] = 0
    labels = expand_labels(seg, int(args.small_threshold**(1/3))) * cells
        
    #nibabel.save(nibabel.Nifti1Image(labels, affine), args.outpath)

    if args.show:
        import napari
        small = cells & (seg == 0)
        expanded = labels * small
        viewer = napari.view_labels(seg)
        viewer.add_labels(labels)
        viewer.add_labels(small)
        viewer.add_labels(expanded)
        napari.run()
        
    return 0

if __name__ == '__main__':
    main()
