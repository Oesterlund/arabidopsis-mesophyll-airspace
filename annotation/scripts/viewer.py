import argparse
import os
import nibabel
import napari

def main():
    parser = argparse.ArgumentParser(description='Viewer for labeled 3d volumes. Uses nifti files.')
    parser.add_argument('imagepath', type=str, help='Path to image')
    parser.add_argument('labelpath', type=str, help='One or more paths to label files', nargs='+', default=[])
    args = parser.parse_args()

    image = nibabel.load(args.imagepath).get_fdata()
    labels = []
    for path in args.labelpath:
        label = nibabel.load(path).get_fdata().astype('int')
        if any((s > t for s,t in zip(label.shape, image.shape))):
            crop = tuple([
                slice(max(0,s-t)//2, max(0,(s-t)//2) + t) for s,t in zip(label.shape, image.shape)
            ])
            print(
                'Cropping mask to fit image',
                f'Mask shape {str(label.shape)}',
                f'Image shape {str(image.shape)}',
                f'Crop {str(crop)}',
                sep='\n'
            )
            label = label[crop]
        labels.append(label)
        
    viewer = napari.Viewer(title='Viewer for labeled 3d voluems. Changes are NOT saved on exit')
    viewer.add_image(image)
    for label in labels:
        viewer.add_labels(label)
    napari.run()
    return 0


if __name__ == '__main__':
    main()
