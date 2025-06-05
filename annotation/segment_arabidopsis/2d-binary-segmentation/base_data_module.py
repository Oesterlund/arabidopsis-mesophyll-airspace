'''See class BaseDataModule'''
import pandas as pd
import pytorch_lightning as pl
import torchio as tio
from torch.utils.data import DataLoader

# TODO: add unlabeled-train as a possible dataset

__all__ = [
    'BaseDataModule'
]

class BaseDataModule(pl.LightningDataModule):
    '''Base class for data modules
    Subclasses should implement the remaining LightningDataModule key methods, see documentation for
    LightningDataModule. 

    Additionaly, subclasses must implement the following methods
        create_subject(row : NamedTuple) -> torchio.Subject
    that create a TorchIO subject from a row in the data_info file.
    For example,

        def create_subject(self, row):
            subject_kwargs = {
                'image' : tio.ScalarImage(row.image_path),
                'fov' : tio.LabelMap(row.fov_path),
                'annotation' : tio.LabelMap(row.annotation_path),
                'weight' : tio.LabelMap(row.weight_path),
            }
            return torchio.Subject(**subject_kwargs)   
    '''
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 data_info_path,
                 batch_size,
                 transforms=None,
                 num_workers=0,
                 generate_patches=False,
                 patch_size=(128,128),
                 queue_length=300,
                 samples_per_volume=10,
                 shuffle_patches=True,
                 patch_sampling_label_name=None,
                 patch_sampling_prob_name=None,
                 extra_data_loader_kwargs=None,
                 verbose=False,
                 ):
        '''
        Parameters
        ----------       
         data_info_path : str
            csv file where each row corresponds to one sample. Must contain a `dataset` column with
            values in {train, validation, test, predict}

        batch_size : int
          Batch size

        transforms : dict of callables, optional
          Dictionary of transforms to apply to each dataset. Example: If
            transforms = {'train' : <some-transform>}
          then <some-transform> will be applied to the "train" dataset and all other datasets will
          not be transformed.

        num_workers : int, optional
          How many workers to use for generating data          

        generate_patches : bool, optional
          If True, sample patches from the images instead of generating full images.
          The following parameters are used to control the patch generation:
            patch_size, queue_length, samples_per_volume, shuffle_patches
          See https://torchio.readthedocs.io/patches/patch_training.html for more info.
          
        patch_size : tuple of int, optional
          Size of generated patches. Ignored unless generate_patches = True

        queue_length : int, optional
          Maximum number of patches to hold in the queue.

        samples_per_volume : int, optional
          Number of samples per image

        shuffle_patches : bool, optional
          If True, the queue will be shuffled before batches are generated from it

        patch_sampling_label_name : str, optional
          If not None and generate_patches is True, then this should match a label mask used to
          control sampling of patches. 

        patch_sampling_prob_name : str, optional
          If not None and generate_patches is True, then this should match a probability mask used
          to control sampling of patches. 

        extra_data_loader_kwargs : dict, optional
          Extrax arguments to pass to the dataloader, e.g. pin_memory

        Notes
        -----
        If a patch-based pipeline is used and all patches from an image should be in the same batch,
        then batch_size, queue_length, samples_per_volume and shuffle_patches need to be set
        as follows
          shuffle_patches must be False.
          batch_size and queue_length must be multiples of samples_per_volume.

        For example,
          shuffle_patches = False
          batch_size = 16
          samples_per_volume = 8
          queue_length = 64
        Then each batch will contain patches from two subjects, the first 8 from one subject and the
        last 8 from the other subject.
        For details regarding patch sampling, see 
        https://torchio.readthedocs.io/patches/patch_training.html        
        '''
        # pylint: disable=too-many-arguments
        super().__init__()
        self.data_info_path = data_info_path
        self.batch_size = batch_size
        self.transforms = {} if transforms is None else transforms
        self.num_workers = num_workers
        self.generate_patches = generate_patches
        self.patch_size = patch_size
        self.queue_length = queue_length
        self.samples_per_volume = samples_per_volume
        self.shuffle_patches = shuffle_patches
        self.patch_sampling_label_name = patch_sampling_label_name
        self.patch_sampling_prob_name = patch_sampling_prob_name
        self.extra_data_loader_kwargs = \
            {} if extra_data_loader_kwargs is None else extra_data_loader_kwargs
        self.verbose = verbose
        self.datasets = {}

    def create_subject(self, row):
        '''
        Parameters
        ----------
        row : NamedTuple
          Row from data_info_path

        Returns
        -------
        subject : torchio.Subject
        '''
        raise NotImplementedError('Inheriting classes must implement create_subject')               

        
    def license(self):
        '''License the data is shared under.
        Only implement if there is a clearly specificed license for the original data source.
        '''
        raise NotImplementedError('No license data provided')

    def setup(self, stage=None):
        # pylint: disable=too-many-branches
        data_info = pd.read_csv(self.data_info_path)

        # If we are being called by a lightning Trainer we only make the
        # relevant datasets. Otherwise (stage is None) we make everything,
        if stage == 'fit':
            data_info = data_info[(data_info['dataset'] == 'train') |
                                  (data_info['dataset'] == 'validation')]
        elif stage == 'validate':
            data_info = data_info[data_info['dataset'] == 'validation']
        elif stage == 'test':
            data_info = data_info[data_info['dataset'] == 'test']
        elif stage == 'predict':
            data_info = data_info[data_info['dataset'] == 'predict']
        
        # Now we can create the actual samples
        samples = {
            'train' : [],
            'validation' : [],
            'test' : [],
            'predict' : []
        }
        for row in data_info.itertuples():            
            try:
                samples[row.dataset].append(self.create_subject(row))
            except KeyError:
                if self.verbose:
                    print(
                        f'Unknown dataset {row.dataset}. Expected train, validation, test, predict'
                    )
                continue
            
        for ds_name in ['train', 'validation', 'test', 'predict']:
            if len(samples[ds_name]) > 0:
                transform = self.transforms[ds_name] if ds_name in self.transforms else None
                self.datasets[ds_name] = tio.SubjectsDataset(samples[ds_name],transform=transform)
                
    def _get_dataloader(self, ds_name, shuffle=False):
        if not ds_name in self.datasets:
            raise KeyError(f'No {ds_name} data available. Ensure that at least one row in '
                           f'"{self.data_info_path}" has dataset = {ds_name}.\n'
                           'Did you forget to call prepare_data() or setup()?')
        if self.generate_patches:
            return self._get_patch_dataloader(ds_name, shuffle)
        return self._get_fullimage_dataloader(ds_name, shuffle)

    
    def _get_fullimage_dataloader(self, ds_name, shuffle=False):
        return DataLoader(
            self.datasets[ds_name],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            **self.extra_data_loader_kwargs
        )

    
    def _get_patch_dataloader(self, ds_name, shuffle=False):
        queue = self._get_patch_dataset(ds_name, shuffle)
        return DataLoader(
            queue,
            batch_size=self.batch_size,
            num_workers=0,
            **self.extra_data_loader_kwargs
        )    
        
    def train_dataloader(self, shuffle=True):
        return self._get_dataloader('train', shuffle)
    
    def val_dataloader(self, shuffle=False):
        return self._get_dataloader('validation', shuffle)

    def test_dataloader(self, shuffle=False):
        return self._get_dataloader('test', shuffle)

    def predict_dataloader(self, shuffle=False):
        return self._get_dataloader('predict', shuffle)

    
    def get_dataset(self, ds_name, shuffle=False):
        '''
        shuffle : bool
          Control of subjects are shuffled after patch sampling.
          Ignored if self.generate_patches == False.
        '''
        if not ds_name in self.datasets:
            raise ValueError(f'No {ds_name} data available. Ensure that at least one row in '
                             f'"{self.data_info_path}" has dataset = {ds_name}.\n'
                             'Did you forget to call prepare_data() or setup()?')
        if self.generate_patches:
            return self._get_patch_dataset(ds_name, shuffle)
        return self._get_fullimage_dataset(ds_name)

    def _get_fullimage_dataset(self, ds_name):
        return self.datasets[ds_name]

    def _get_patch_dataset(self, ds_name, shuffle=False):
        if self.patch_sampling_label_name is not None:
            sampler = tio.data.LabelSampler(
                self.patch_size,
                self.patch_sampling_label_name,
            )
        elif self.patch_sampling_prob_name is not None:
            sampler = tio.data.WeightedSampler(
                self.patch_size,
                self.patch_sampling_prob_name,
            )
        else:
            sampler = tio.data.UniformSampler(self.patch_size)
        queue = tio.Queue(
            self.datasets[ds_name],
            self.queue_length,
            self.samples_per_volume,
            sampler,
            num_workers=self.num_workers,
            shuffle_subjects=shuffle,
            shuffle_patches=self.shuffle_patches,
        )
        return queue
    
    def train_subjects(self):
        '''get the train subjects dataset'''
        return self._get_subjects('train')

    def val_subjects(self):
        '''get the validation subjects dataset'''
        return self._get_subjects('validation')

    def test_subjects(self):
        '''get the test subjects dataset'''
        return self._get_subjects('test')

    def predict_subjects(self):
        '''get the prediction subjects dataset'''
        return self._get_subjects('predict')
    
    def _get_subjects(self, ds_name):
        if not ds_name in self.datasets:
            raise ValueError(f'No {ds_name} data available. Ensure that at least one row in '
                             f'"{self.data_info_path}" has dataset = {ds_name}.\n'
                             'Did you forget to call prepare_data() or setup()?')
        return self.datasets[ds_name]

    
    def full_grid_inference(
            self,
            model,
            ds_name,
            device,
            patch_size=None,
            patch_overlap=None,
            patch_batch_size=None,
            return_subject=False
    ):
        # TODO: Rewrite to just return a sampler.
        # pylint: disable=too-many-arguments
        '''For patch based model. Aggregate patch predictions to get full volume predictions
        
        Parameters
        ----------
        model : torch model
          must have a method `predict_batch(patches_batch, device) -> patch_predictions`

        ds_name : str
          Dataset to apply the model on. This is to simplify only applying full inference on a small
          part of the data.

        device : torch.device or int
          Device used for predictions

        patch_size : tuple of int, optional
          Size of patches used for prediction. If None use patch size from construction
        
        patch_overlap : tuple of int, optional
          Overlap of patches along each axis. If None use half of patch_size

        patch_batch_size : int, optional
          Batch size for patch predictions. If None use batch size from contruction

        return_subject : bool, optional
          If True yield a tuple of (prediction, subject), otherwise only yield
          prediction

        Returns
        -------
        Either
          generator of torch.tensor
            Generator that yields single subject predictions
        Or
          generator of (torch.tensor, tio.Subject)
            Generator that yields single subject predictions with the corresponding tio.Subject

        '''
        model = model.to(device)
        if patch_size is None:
            patch_size = self.patch_size
        if patch_overlap is None:
            patch_overlap = tuple([s//2 for s in patch_size])
        if patch_batch_size is None:
            patch_batch_size = self.batch_size

        for subject in self.datasets[ds_name]:
            grid_sampler = tio.inference.GridSampler(
                subject,
                patch_size,
                patch_overlap,
                padding_mode=0
            )
            patch_loader = DataLoader(grid_sampler, batch_size=patch_batch_size)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')
            for patches_batch in patch_loader:
                predictions = model.predict_batch(patches_batch, device).cpu()
                locations = patches_batch[tio.LOCATION]
                aggregator.add_batch(predictions, locations)
            if return_subject:
                yield aggregator.get_output_tensor(), subject
            else:
                yield aggregator.get_output_tensor()


    def full_grid_feature_extraction(
            self,
            model,
            ds_name,
            device,
            patch_size=None,
            patch_overlap=None,
            patch_batch_size=None,
            return_paths=False
    ):
        # pylint: disable=too-many-arguments
        '''For patch based model. Aggregate patch feature maps to get full volume feature maps
        
        Parameters
        ----------
        model : torch model
          must have a method `extract_feature(patches_batch, device) -> feature_maps`

        ds_name : str
          Dataset to apply the model on. This is to simplify only applying full inference on a small
          part of the data.

        device : torch.device or int
          Device used for predictions

        patch_size : tuple of int, optional
          Size of patches used for prediction. If None use patch size from construction
        
        patch_overlap : tuple of int, optional
          Overlap of patches along each axis. If None use half of patch_size

        patch_batch_size : int, optional
          Batch size for patch predictions. If None use batch size from contruction

        return_paths : bool, optional
          If True yield a tuple of (prediction, path-to-input-image), otherwise only yield
          prediction

        Returns
        -------
        Either
          generator of torch.tensor 
            Generator that yields single subject feature maps
        Or
          generator of (torch.tensor, str)
            Generator that yields tuple of (single subject feature maps, path to input image)

        '''
        model = model.to(device)
        if patch_size is None:
            patch_size = self.patch_size
        if patch_overlap is None:
            patch_overlap = tuple([s//2 for s in patch_size])
        if patch_batch_size is None:
            patch_batch_size = self.batch_size

        for subject in self.datasets[ds_name]:
            grid_sampler = tio.inference.GridSampler(
                subject,
                patch_size,
                patch_overlap,
                padding_mode=0
            )
            patch_loader = DataLoader(grid_sampler, batch_size=patch_batch_size)
            aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode='hann')
            for patches_batch in patch_loader:
                features = model.extract_feature(patches_batch, device).cpu()
                locations = patches_batch[tio.LOCATION]
                aggregator.add_batch(features, locations)
            if return_paths:
                yield aggregator.get_output_tensor(), subject['image'][tio.PATH]
            else:
                yield aggregator.get_output_tensor()
                
