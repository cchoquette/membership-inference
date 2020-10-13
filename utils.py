import numpy as np
import scipy.ndimage.interpolation as interpolation
import tensorflow as tf
import csv


def create_rotates(r):
  """Creates vector of rotation degrees compatible with scipy' rotate.

  Args:
    r: param r for rotation augmentation attack. Defines max rotation by +/-r. Leads to 2*r+1 total images per sample.

  Returns: vector of rotation degrees compatible with scipy' rotate.

  """
  if r is None:
    return None
  if r == 1:
    return [0.0]
  # rotates = [360. / r * i for i in range(r)]
  rotates = np.linspace(-r, r, (r * 2 + 1))
  return rotates


def create_translates(d):
  """Creates vector of translation displacements compatible with scipy' translate.

  Args:
    d: param d for translation augmentation attack. Defines max displacement by d. Leads to 4*d+1 total images per sample.

  Returns: vector of translation displacements compatible with scipy' translate.
  """
  if d is None:
    return None
  def all_shifts(mshift):
    if mshift == 0:
      return [(0, 0, 0, 0)]
    all_pairs = []
    start = (0, mshift, 0, 0)
    end = (0, mshift, 0, 0)
    vdir = -1
    hdir = -1
    first_time = True
    while (start[1] != end[1] or start[2] != end[2]) or first_time:
      all_pairs.append(start)
      start = (0, start[1] + vdir, start[2] + hdir, 0)
      if abs(start[1]) == mshift:
        vdir *= -1
      if abs(start[2]) == mshift:
        hdir *= -1
      first_time = False
    all_pairs = [(0, 0, 0, 0)] + all_pairs  # add no shift
    return all_pairs

  translates = all_shifts(d)
  return translates


def apply_augment(ds, augment, type_):
  """Applies an augmentation from create_rotates or create_translates.

  Args:
    ds: tuple of (images, labels) describing a datset. Images should be 4D of (N,H,W,C) where N is total images.
    augment: the augment to apply. (one element from augments returned by create_rotates/translates)
    type_: attack type, either 'd' or 'r'

  Returns:

  """
  if type_ == 'd':
    ds = (interpolation.shift(ds[0], augment, mode='nearest'), ds[1])
  else:
    ds = (interpolation.rotate(ds[0], augment, (1, 2), reshape=False), ds[1])
  return ds


def softmax(x):
  """Transform logits to softmax.

  Args:
    x: vector or matrix logits.

  Returns: matrix of logits.

  """
  if len(np.shape(x)) == 1:
    x = x[np.newaxis, :]
  s = np.max(x, axis=1)[:, np.newaxis]
  e_x = np.exp(x - s)
  div = np.sum(e_x, axis=1)
  div = div[:, np.newaxis]
  return e_x / div


def convert_data(x, y, ndata, skip, dataset):
  x = x.astype(np.float32)
  start = skip * ndata
  end = start + ndata
  x = x[start:end]
  y = y[start:end]
  # y = tf.keras.utils.to_categorical(y, num_classes=10)
  if dataset == 'cifar10' or dataset == 'mnist' or dataset == 'cifar100':
    x /= 255.
  p = np.random.permutation(len(x))
  x = x[p]
  y = y[p]
  return x, y


def load_data(name):
  if name == 'cifar10' or name == 'mnist' or name == 'cifar100':
    (x_train, y_train), (x_test, y_test) = getattr(tf.keras.datasets, name).load_data()
    input_dim = (32, 32, 3) if name != 'mnist' else (28, 28, 1)
    if name == 'mnist':
      x_train = x_train[:, :, None]
      x_test = x_test[:, :, None]
    n_classes = 10 if name != 'cifar100' else 100
  elif name == 'adult':
    data = np.load('data/adult/adult.npz')
    x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']
    all_x = np.concatenate([x_train, x_test], axis=0)
    all_y = np.concatenate([y_train, y_test], axis=0)
    indices = np.arange(len(all_y))
    p = np.random.permutation(indices)
    x_train, y_train = all_x[p[:10000]], all_y[p[:10000]]
    x_test, y_test = all_x[p[10000:]], all_y[p[10000:]]
    input_dim = 107
    n_classes = 2
  elif name == 'purchase':
    x, y = [], []
    with open('data/purchase', 'r') as infile:
      reader = csv.reader(infile)
      for line in reader:
        y.append(int(line[0]))
        x.append([int(x) for x in line[1:]])
      x = np.array(x)
      y = (np.array(y) - 1).reshape((-1, 1))
      indices = np.arange(len(y))
      p = np.random.permutation(indices)
      x_train, y_train = x[p[:10000]], y[p[:10000]]
      x_test, y_test = x[p[10000:]], y[p[10000:]]
      input_dim = 600
      n_classes = 100
  elif name == 'texas':
    x, y = [], []
    with open('data/texas/100/feats', 'r') as infile:
      reader = csv.reader(infile)
      for line in reader:
        x.append([int(x) for x in line[1:]])
      x = np.array(x)
    with open('data/texas/100/labels', 'r') as infile:
      reader = csv.reader(infile)
      for line in reader:
        y.append(int(line[0]))
      y = (np.array(y) - 1).reshape((-1, 1))
      indices = np.arange(len(y))
      p = np.random.permutation(indices)
      x_train, y_train = x[p[:10000]], y[p[:10000]]
      x_test, y_test = x[p[10000:]], y[p[10000:]]
      input_dim = 6168
      n_classes = 100
  elif name == 'location':
    x, y = [], []
    with open('data/location', 'r') as infile:
      reader = csv.reader(infile)
      for line in reader:
        y.append(int(line[0]))
        x.append([int(x) for x in line[1:]])
      x = np.array(x)
      y = (np.array(y) - 1).reshape((-1, 1))
      indices = np.arange(len(y))
      p = np.random.permutation(indices)
      x_train, y_train = x[p[:1600]], y[p[:1600]]
      x_test, y_test = x[p[1600:]], y[p[1600:]]
      input_dim = 446
      n_classes = 30
  else:
    raise ValueError(f"dataset: {name} not valid.")
  print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
  return (x_train, y_train), (x_test, y_test), input_dim, n_classes


def get_data(dataset, ndata):
  (x_train, y_train), (x_test, y_test), input_dim, n_classes = load_data(dataset)  # load data

  if dataset == 'cifar10' or dataset == 'cifar100' or dataset == 'mnist':
    all_data = (np.concatenate([x_train, x_test], 0),
                np.concatenate([y_train, y_test], 0))
    target_train_set = convert_data(*all_data, ndata, 0, dataset)
    target_test_set = convert_data(*all_data, ndata, 1, dataset)

    source_train_set = convert_data(*all_data, ndata, 2, dataset)
    source_test_set = convert_data(*all_data, ndata, 3, dataset)
  elif dataset == 'texas' or dataset == 'purchase' or dataset == 'adult' or dataset == 'location':
    target_train_set = (x_train, y_train)
    target_test_set = convert_data(*(x_test, y_test), ndata, 1, dataset)

    source_train_set = convert_data(*(x_test, y_test), ndata, 2, dataset)
    source_test_set = convert_data(*(x_test, y_test), ndata, 3, dataset)
  else:
    raise ValueError(f"dataset: {dataset} not supported.")
  return target_train_set, target_test_set, source_train_set, source_test_set, input_dim, n_classes
