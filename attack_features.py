import numpy as np
import scipy.ndimage.interpolation as interpolation
import tensorflow as tf
from utils import apply_augment, create_rotates, create_translates, softmax


def check_correct(ds, predictions):
  """Used for augmentation MI attack to check if each image was correctly classified using label-only access.

  Args:
    ds: tuple of (images, labels) where images are (N, H, W, C).
    predictions: predictions from model.

  Returns: 1 if correct, 0 if incorrect for each sample.

  """
  return np.equal(ds[1].flatten(), np.argmax(predictions, axis=1)).squeeze()


def augmentation_attack(model, train_set, test_set, max_samples, attack_type='d', augment_kwarg=1, batch=100):
  if attack_type == 'r':
    augments = create_rotates(augment_kwarg)
  elif attack_type == 'd':
    augments = create_translates(augment_kwarg)
  else:
    raise ValueError(f"attack type_: {attack_type} is not valid.")
  m = np.concatenate([np.ones(max_samples),
                      np.zeros(max_samples)], axis=0)
  attack_in = np.zeros((max_samples, len(augments)))
  attack_out = np.zeros((max_samples, len(augments)))
  for i, augment in enumerate(augments):
    train_augment = apply_augment(train_set, augment, attack_type)
    test_augment = apply_augment(test_set, augment, attack_type)
    train_ds = tf.data.Dataset.from_tensor_slices(train_augment).batch(batch)
    test_ds = tf.data.Dataset.from_tensor_slices(test_augment).batch(batch)
    in_ = softmax(model.predict(train_ds))
    out_ = softmax(model.predict(test_ds))
    attack_in[:, i] = check_correct(train_set, in_)[:max_samples]
    attack_out[:, i] = check_correct(test_set, out_)[:max_samples]
  attack_set = (np.concatenate([attack_in, attack_out], 0),
                np.concatenate([train_set[1], test_set[1]], 0),
                m)
  return attack_set

