'''See class NetWrapper'''
import torch
import pytorch_lightning as pl
import torchio as tio

__all__ = [
    'NetWrapper'
]


class NetWrapper(pl.LightningModule):
    '''Simple pytorch lightning wrapper for an net '''
    def __init__(self, net, loss, learning_rate):
        super().__init__()
        self.net = net
        self.loss = loss
        self.learning_rate = learning_rate
        self.save_hyperparameters() 

    def forward(self, x):
        # pylint: disable=arguments-differ
        return self.net(x)

    def prepare_batch(self, batch, return_filename=False):
        '''Assumes batch is a torchio dataset'''
        images = batch['image'][tio.DATA].squeeze(-1)
        labels  = batch['label'][tio.DATA].squeeze(-1)
        if return_filename:
            return images, labels, batch['image']['path']
        return images, labels

    def training_step(self, batch, batch_idx, *args, **kwargs):
        # pylint: disable=arguments-differ
        images, labels = self.prepare_batch(batch)
        preds = self(images)
        loss = self.loss(preds, labels)
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=images.shape[0])
        #if self.current_epoch % 5 == 0 and batch_idx == 0:
        if batch_idx == 0:
            self.logger.experiment.add_images(
                f'train_prediction',
                torch.sigmoid(preds.mT),
                self.current_epoch
            )
            self.logger.experiment.add_images(
                f'train_labels',
                (255*labels.mT),
                self.current_epoch
            )
            self.logger.experiment.add_images(
                f'train_images',
                images.mT,
                self.current_epoch
            )
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        # pylint: disable=arguments-differ
        images, labels = self.prepare_batch(batch)
        preds = self(images)
        loss = self.loss(preds, labels)
        self.log('val_loss', loss, batch_size=images.shape[0])
        #if self.current_epoch % 5 == 0 and batch_idx == 0:
        if batch_idx == 0:            
            self.logger.experiment.add_images(
                f'val_prediction',
                torch.sigmoid(preds.mT),
                self.current_epoch
            )
            self.logger.experiment.add_images(
                f'val_labels',
                (255*labels.mT),
                self.current_epoch
            )
            self.logger.experiment.add_images(
                f'val_images',
                images.mT,
                self.current_epoch
            )
        return loss

    def predict_step(self, batch, *args, return_filename=False, device='cuda', **kwargs):
        # pylint: disable=arguments-differ, unused-argument
        images = batch['image'][tio.DATA].squeeze(-1).to(device)
        #print(images.device)
        predictions = torch.sigmoid(self(images))
        if return_filename:
            return predictions, images, batch['image']['path']
        return predictions, images
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
