import argparse
import os
import numpy as np
import nibabel
import napari

def main():
    parser = argparse.ArgumentParser(description='Viewer for labeled 3d volumes. Uses nifti files.')
    parser.add_argument('imagepath', type=str, help='Path to image')
    parser.add_argument('labelpath', type=str, help='One or more paths to label files', nargs='*', default=[])
    args = parser.parse_args()

    image = np.asanyarray(nibabel.load(args.imagepath).dataobj)
    labels = []
    for path in args.labelpath:
        label = np.asanyarray(nibabel.load(path).dataobj, dtype=int)
        labels.append(label)
        
    viewer = napari.Viewer(title='Viewer for labeled 3d volumes. Changes are NOT saved on exit')
    viewer.add_image(image)
    for label in labels:
        viewer.add_labels(label)
    napari.run()
    return 0


if __name__ == '__main__':
    main()
