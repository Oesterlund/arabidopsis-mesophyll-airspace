import argparse
import os
import nibabel
import numpy as np
import scipy.ndimage as ndi
import napari
        
def main():    
    parser = argparse.ArgumentParser()
    parser.add_argument('semantic_segpath', type=str, help='Path to semantic segmentation')
    parser.add_argument('outpath', type=str, help='Path to segmentation in.')
    parser.add_argument('--epidermis-id', type=int, help='Value corresponding to epidermis.', default=4)
    parser.add_argument('--outside-id', type=int, help='Value corresponding to outside.', default=2)
    parser.add_argument('--sigma', type=float, help='Scale of Gaussian used for gradient computation.', default=1)
    parser.add_argument('--downsample', action='store_true')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    outdir = os.path.dirname(args.outpath)
    if outdir != '':
        os.makedirs(outdir, exist_ok=True)
        
    seg = nibabel.load(args.semantic_segpath)
    affine = seg.affine
    seg = np.asanyarray(seg.dataobj)
    if args.downsample:
        seg = seg[::4,::4,::4]

    labels,_ = ndi.label(seg == args.epidermis_id)

    # Find the two largests components
    _, counts = np.unique(labels, return_counts=True)
    res = np.argpartition(counts[1:], -3) #[-2,-1])

    c1 = labels != 1 + res[-2]
    c2 = labels != 1 + res[-1]

    print('Distance transforms')
    d1 = ndi.distance_transform_edt(c1)
    d2 = ndi.distance_transform_edt(c2)
    
    print('Gradients')
    g1_y = ndi.gaussian_filter(d1, order=[0,1,0], sigma=args.sigma)
    g1_x = ndi.gaussian_filter(d1, order=[0,0,1], sigma=args.sigma)
    g2_y = ndi.gaussian_filter(d2, order=[0,1,0], sigma=args.sigma)
    g2_x = ndi.gaussian_filter(d2, order=[0,0,1], sigma=args.sigma)

    angles = g1_y*g2_y + g1_x*g2_x
    outside = ((angles > 0) & c1 & c2).astype('uint8')

    # Find the two largests components of outside
    labels,_ = ndi.label(outside)
    _, counts = np.unique(labels, return_counts=True)
    res = np.argpartition(counts[1:], -3)
    outside = (labels == (1 + res[-2])) | (labels == (1 + res[-1]))

    if args.show:
        viewer = napari.view_labels(seg)
        viewer.add_labels(outside)
        napari.run()

    seg[outside == 1] = args.outside_id
    nibabel.save(nibabel.Nifti1Image(seg, affine), args.outpath)

    return 0

if __name__ == '__main__':
    main()
