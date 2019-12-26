import numpy as np
import torch
from skimage import transform
import os


class PtDataset(torch.utils.data.Dataset):
    """
    dataset of image files of the form 
       stuff<number>_trans.pt
       stuff<number>_density.pt
    """

    def __init__(self, data_dir, index_slice=None, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # f_trans = [f for f in os.listdir(data_dir)
        #            if f.endswith('trans_1d.pt')]
        f_trans = [f for f in os.listdir(data_dir)
                   if f.startswith('input')]
        f_trans.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        # f_density = [f for f in os.listdir(data_dir)
        #              if f.endswith('density_1d.pt')]
        f_density = [f for f in os.listdir(data_dir)
                     if f.startswith('label')]
        f_density.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        if index_slice:
            f_trans = f_trans[index_slice]
            f_density = f_density[index_slice]

        self.names = (f_trans, f_density)

    def __getitem__(self, index):
        # x = torch.load(os.path.join(self.data_dir, self.names[0][index]))
        # y = torch.load(os.path.join(self.data_dir, self.names[1][index]))
        x = np.load(os.path.join(self.data_dir, self.names[0][index]))
        y = np.load(os.path.join(self.data_dir, self.names[1][index]))
        # x = x.to(self.device)
        # y = y.to(self.device)

        # x = np.expand_dims(np.expand_dims(x, axis=1), axis=2)
        # y = np.expand_dims(np.expand_dims(y, axis=1), axis=2)

        data = {'input': x, 'label': y}

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

        input, label = data['input'], data['label']
        input = input.transpose((2, 0, 1)).astype(np.float32)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        return {'input': torch.from_numpy(input), 'label': torch.from_numpy(label)}


class Nomalize(object):
    def __call__(self, data):
        # Nomalize [0, 1] => [-1, 1]

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data

        input, label = data['input'], data['label']
        input = 2 * (input / 255) - 1
        label = 2 * (label / 255) - 1
        return {'input': input, 'label': label}


class RandomFlip(object):
    def __call__(self, data):
        # Random Left or Right Flip

        # for key, value in data:
        #     data[key] = 2 * (value / 255) - 1
        #
        # return data
        input, label = data['input'], data['label']

        if np.random.rand() > 0.5:
            input = np.fliplr(input)
            label = np.fliplr(label)

        # if np.random.rand() > 0.5:
        #     input = np.flipud(input)
        #     label = np.flipud(label)

        return {'input': input, 'label': label}


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
    input, label = data['input'], data['label']

    h, w = input.shape[:2]

    if isinstance(self.output_size, int):
      if h > w:
        new_h, new_w = self.output_size * h / w, self.output_size
      else:
        new_h, new_w = self.output_size, self.output_size * w / h
    else:
      new_h, new_w = self.output_size

    new_h, new_w = int(new_h), int(new_w)

    input = transform.resize(input, (new_h, new_w))
    label = transform.resize(label, (new_h, new_w))

    return {'input': input, 'label': label}


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
    input, label = data['input'], data['label']

    h, w = input.shape[:2]
    new_h, new_w = self.output_size

    top = np.random.randint(0, h - new_h)
    left = np.random.randint(0, w - new_w)

    input = input[top: top + new_h, left: left + new_w]
    label = label[top: top + new_h, left: left + new_w]

    return {'input': input, 'label': label}


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
