from .field import RawField, Merge, ImageDetectionsField, TextField
from .dataset import Sydney,UCM,RSICD
from torch.utils.data import DataLoader as TorchDataLoader

class DataLoader(TorchDataLoader):
    def __init__(self, dataset, *args, **kwargs):
        super(DataLoader, self).__init__(dataset, *args, collate_fn=dataset.collate_fn(), **kwargs)

    def __len__(self):
        return len(self.dataset.examples)//self.batch_size
