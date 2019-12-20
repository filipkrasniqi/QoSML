from torch.utils import data

class Dataset(data.Dataset):
  def __init__(self, input, target, window_size = 8):
        'Initialization'
        self.window_size = window_size
        self.input = input
        self.target = target[window_size-1:]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.input) - self.window_size + 1

  def __getitem__(self, index):
        X = self.input[index:index+self.window_size]
        y = self.target[index]
        return X, y
