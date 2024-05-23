from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import numpy as np


# =============================================================================
class NumpyDataset2D(torch.utils.data.Dataset):
    def __init__(self, numpy_file_path, n_samples, use_cuda):
      self.use_cuda = use_cuda
      self.numpy_file_path = numpy_file_path
      self.n_samples = n_samples

      train_data = np.expand_dims(np.load(self.numpy_file_path), axis=1)[:self.n_samples, :, :, :]
      max_v, min_v = train_data.max(), train_data.min()
      train_data = ((train_data-min_v)/(max_v-min_v))*2 - 1.
      self.x_data = torch.tensor(train_data, dtype=torch.float32)

      if self.use_cuda:
        self.x_data.cuda()
    
    def __len__(self):
      return len(self.x_data)

    def __getitem__(self, idx):
      return self.x_data[idx]


def get_numpy_dataloader2D(numpy_file_path, batch_size=64, n_samples=8000, use_cuda=False):
    ds = NumpyDataset2D(numpy_file_path=numpy_file_path, n_samples=n_samples, use_cuda=use_cuda)
    data_loader = torch.utils.data.DataLoader(dataset=ds, batch_size=batch_size, shuffle=True, )
    return data_loader

