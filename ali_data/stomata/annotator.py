import argparse
import os
import nibabel
import napari
import numpy as np
from skimage.io import imread
import skimage
import tifffile

def main():
    parser = argparse.ArgumentParser(description='Annotation tool for 3d volumes. Uses nifti files.')
    parser.add_argument('imagepath', type=str, help='Path to image to annotate')
    parser.add_argument('outpath', type=str, help='Path to store annotation in. '
                        '(Use .nii.gz extension for compression)')
    parser.add_argument('--annotation', type=str, help='Path to existing annotation file')
    parser.add_argument('--background-mask', type=str, nargs='+', default=[],
                        help='One or more paths to existing segmentation masks to use as background '
                        'in the nitial annotation. Ignored if --annotation is used')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)

    image=imread(args.imagepath)
    #image = nibabel.load(args.imagepath)
    #affine = image.affine
    #image = image.get_fdata()

    if args.annotation is not None and os.path.exists(args.annotation):
        mask = imread(args.annotation).astype('uint8')
    else:
        mask = np.zeros_like(image, dtype='uint8')
        for path in args.background_mask:
            bg = imread(path) > 0
            if any((s > t for s,t in zip(bg.shape, image.shape))):
                crop = tuple([
                    slice(max(0,s-t)//2, max(0,(s-t)//2) + t) for s,t in zip(bg.shape, image.shape)
                ])
                print(
                    'Cropping background mask to fit image',
                    f'Mask shape {str(bg.shape)}',
                    f'Image shape {str(image.shape)}',
                    f'Crop {str(crop)}',
                    sep='\n'
                )
                bg = bg[crop]
            mask[bg] = 2
    try:
        viewer = napari.Viewer(title='Annotator for 3d voluems. Changes are saved on exit')
        viewer.add_image(image)
        viewer.add_labels(mask)
        napari.run()
    except Exception as e:
        print('Napari closed unexpectedly:', e)
    #mask = nibabel.Nifti1Image(mask, affine)
    #nibabel.save(mask, args.outpath)
    #skimage.io.imsave(args.outpath,mask)
    tifffile.imwrite(args.outpath+'.tiff', mask, photometric='minisblack',imagej=True)


    return 0


if __name__ == '__main__':
    main()
