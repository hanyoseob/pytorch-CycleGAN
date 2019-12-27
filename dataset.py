import numpy as np
import torch
from skimage import transform
import matplotlib.pyplot as plt
import os


class Dataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, direction = 'A2B', data_type='float32', index_slice=None, transform=None):
        self.data_dir_a = data_dir + 'A'
        self.data_dir_b = data_dir + 'B'
        self.transform = transform
        self.direction = direction
        self.data_type = data_type

        dataA = [f for f in os.listdir(self.data_dir_a) if f.endswith('.jpg')]
        dataA.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        dataB = [f for f in os.listdir(self.data_dir_b) if f.endswith('.jpg')]
        dataB.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if index_slice:
            dataA = dataA[index_slice]
            dataB = dataB[index_slice]

        self.names = (dataA, dataB)

    def __getitem__(self, index):

        # x = np.load(os.path.join(self.data_dir, self.names[0][index]))
        # y = np.load(os.path.join(self.data_dir, self.names[1][index]))

        dataA = plt.imread(os.path.join(self.data_dir_a, self.names[0][index])).squeeze()
        dataB = plt.imread(os.path.join(self.data_dir_b, self.names[1][index])).squeeze()

        if self.data_type == 'float32':
            dataA = dataA.astype(np.float32)
            dataB = dataB.astype(np.float32)

        if len(dataA.shape) == 2:
            dataA = np.expand_dims(dataA, axis=3)
            dataA = np.tile(dataA, (1, 1, 3))
        if len(dataB.shape) == 2:
            dataB = np.expand_dims(dataB, axis=3)
            dataB = np.tile(dataB, (1, 1, 3))

        if self.direction == 'A2B':
            data = {'dataA': dataA, 'dataB': dataB}
        else:
            data = {'dataA': dataB, 'dataB': dataA}

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.names[0])


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = torch.from_numpy(value.transpose((2, 0, 1)))
        #
        # return data

        dataA, dataB = data['dataA'], data['dataB']

        dataA = dataA.transpose((2, 0, 1)).astype(np.float32)
        dataB = dataB.transpose((2, 0, 1)).astype(np.float32)
        return {'dataA': torch.from_numpy(dataA), 'dataB': torch.from_numpy(dataB)}


class Nomalize(object):
    def __call__(self, data):
        # Nomalize [0, 1] => [-1, 1]

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data

        dataA, dataB = data['dataA'], data['dataB']
        dataA = 2 * (dataA / 255) - 1
        dataB = 2 * (dataB / 255) - 1
        return {'dataA': dataA, 'dataB': dataB}


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        dataA, dataB = data['dataA'], data['dataB']

        if np.random.rand() > 0.5:
            dataA = np.fliplr(dataA)
            dataB = np.fliplr(dataB)

        # if np.random.rand() > 0.5:
        #     dataA = np.flipud(dataA)
        #     dataB = np.flipud(dataB)

        return {'dataA': dataA, 'dataB': dataB}


class Rescale(object):
  """Rescale the image in a sample to a given size

  Args:
    output_size (tuple or int): Desired output size.
                                If tuple, output is matched to output_size.
                                If int, smaller of image edges is matched
                                to output_size keeping aspect ratio the same.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    self.output_size = output_size

  def __call__(self, data):
    dataA, dataB = data['dataA'], data['dataB']

    h, w = dataA.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    dataA = transform.resize(dataA, (new_h, new_w))
    dataB = transform.resize(dataB, (new_h, new_w))

    return {'dataA': dataA, 'dataB': dataB}


class RandomCrop(object):
  """Crop randomly the image in a sample

  Args:
    output_size (tuple or int): Desired output size.
                                If int, square crop is made.
  """

  def __init__(self, output_size):
    assert isinstance(output_size, (int, tuple))
    if isinstance(output_size, int):
      self.output_size = (output_size, output_size)
    else:
      assert len(output_size) == 2
      self.output_size = output_size

  def __call__(self, data):
    dataA, dataB = data['dataA'], data['dataB']

    h, w = dataA.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    dataA = dataA[top: top + new_h, left: left + new_w]
    dataB = dataB[top: top + new_h, left: left + new_w]

    return {'dataA': dataA, 'dataB': dataB}


class ToNumpy(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        # Swap color axis because numpy image: H x W x C
        #                         torch image: C x H x W

        # for key, value in data:
        #     data[key] = value.transpose((2, 0, 1)).numpy()
        #
        # return data

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 1)

        # input, label = data['input'], data['label']
        # input = input.transpose((2, 0, 1))
        # label = label.transpose((2, 0, 1))
        # return {'input': input.detach().numpy(), 'label': label.detach().numpy()}


class Denomalize(object):
    def __call__(self, data):
        # Denomalize [-1, 1] => [0, 1]

        # for key, value in data:
        #     data[key] = (value + 1) / 2 * 255
        #
        # return data

        return (data + 1) / 2

        # input, label = data['input'], data['label']
        # input = (input + 1) / 2 * 255
        # label = (label + 1) / 2 * 255
        # return {'input': input, 'label': label}
