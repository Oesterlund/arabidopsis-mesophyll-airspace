import argparse
import os
import nibabel
import numpy as np

def main():
    description = '''Remove instances that touch the boundary of the image'''
    parser = argparse.ArgumentParser(description)
    parser.add_argument('inpath', type=str, help='Path to instance segmentation')
    parser.add_argument('outpath', type=str, help='Path to store post processed instance segmentation')
    parser.add_argument('--boundary-thickness', type=int, required=True)
    parser.add_argument('--boundary-is-cylinder', action='store_true', help='If set, then the boundary is from the largest cylinder that can fit inside the image')
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()
    
    outdir = os.path.dirname(args.outpath)
    if outdir != '':
        os.makedirs(outdir, exist_ok=True)
    
    seg = nibabel.load(args.inpath)
    affine = seg.affine
    seg = np.asanyarray(seg.dataobj)

    t = args.boundary_thickness if 0 < args.boundary_thickness < (min(seg.shape)// 2 - 1) else 1    
    if args.boundary_is_cylinder:
        mask = np.zeros(seg.shape, dtype=bool)
        c1 = seg.shape[1]//2
        c2 = seg.shape[2]//2
        r2 = (min(c1, c2) - t)**2
        for i in range(seg.shape[1]):
            i2 = (i-c1)**2
            for j in range(seg.shape[2]):
                j2 = (j-c2)**2
                if i2 + j2 <= r2:
                    mask[t:-(t+1),i,j] = 1
        boundary = ~mask
    else:
        boundary = np.ones_like(seg, dtype=bool)
        boundary[t:-t,t:-t,t:-t] = 0

    labels_to_remove = np.unique(seg[boundary])
    labels_to_remove = labels_to_remove[labels_to_remove != 0]
    seg[np.isin(seg, labels_to_remove)] = 0
    
    nibabel.save(nibabel.Nifti1Image(seg, affine), args.outpath)

    if args.show:
        import napari
        napari.view_labels(seg)
        napari.run()

if __name__ == '__main__':
    main()
