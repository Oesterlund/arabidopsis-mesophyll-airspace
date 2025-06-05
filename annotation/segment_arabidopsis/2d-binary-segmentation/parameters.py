import torchio as tio

def get_params(version, predict=False):
    if version == 0:
        params = {
            'data_dir' : '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/2d-binary-segmentation/data/abaxial/',
            'data_info_path' : 'data-info-00.csv',
            'batch_size' :  1,
            'train_ratio' : 0.9,
            'val_ratio' : 0.1,
            'transforms' : _get_transforms(version, predict)
        }
    if version == 1:
        params = {
            'data_dir' : '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/2d-binary-segmentation/data/abaxial/',
            'data_info_path' : '/home/isabella/Documents/PLEN/x-ray/annotation/segment_arabidopsis/2d-binary-segmentation/data-info-00.csv',
            'batch_size' :  1
            }
    return params


def _get_transforms(version, predict):
    augment = [
        tio.RandomFlip([0,1], flip_probability=0.5)
    ]
    preprocess = [
        tio.transforms.RescaleIntensity(out_min_max=(0,1), in_min_max=(0,255)),
        tio.RemapLabels({255:1}), #, 100:2, 150:3, 200:4}),
        #tio.Crop((10,10,10,10,0,0)),
        #tio.OneHot(5),
    ]
        
    transforms_train = tio.transforms.Compose(preprocess + augment)
    transforms_others = tio.transforms.Compose(preprocess)
    
    return {
        'train' : transforms_train,
        'validation' : transforms_others,
        'test' : transforms_others,
        'predict' : transforms_others,
    }
