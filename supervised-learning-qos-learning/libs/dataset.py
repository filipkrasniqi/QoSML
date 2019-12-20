from torch.utils import data

class Dataset(data.Dataset):
  def __init__(self, input, target):
        'Initialization'
        self.input = input
        self.target = target

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.input)

  def __getitem__(self, index):
        X = self.input[index]
        y = self.target[index]
        return X, y
