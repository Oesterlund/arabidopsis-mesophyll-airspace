import argparse
import os
import nibabel
import numpy as np

def main():
    description = '''Relabel to have consecutive labels'''
    parser = argparse.ArgumentParser(description)
    parser.add_argument('inpath', type=str, help='Path to instance segmentation')
    parser.add_argument('outpath', type=str, help='Path to store post-processed segmentation')
    args = parser.parse_args()
    
    outdir = os.path.dirname(args.outpath)
    if outdir != '':
        os.makedirs(outdir, exist_ok=True)
    
    seg = nibabel.load(args.inpath)
    affine = seg.affine
    seg = np.asanyarray(seg.dataobj)
    n_labels = np.max(seg)
    
    # Relabel so we still have consecutive labels
    # Should take O(n) on evarege, with O(n^2) worst case.
    new_labels = { old_label : new_label for new_label, old_label in enumerate(np.unique(labels)) }
    for i in range(labels.size):
        old_label = labels.flat[i]
        if old_label != 0:
            labels.flat[i] = new_labels[old_label]    
    
    nibabel.save(nibabel.Nifti1Image(labels, affine), args.outpath)        
    return 0

if __name__ == '__main__':
    main()
