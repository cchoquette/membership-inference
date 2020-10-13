import tensorflow.compat.v1 as tf
from tensorflow.python.keras import backend as K
from cleverhans.model import CallableModelWrapper
from cleverhans.attacks import CarliniWagnerL2, HopSkipJumpAttack  # need to install latest cleverhans (not via pip) for HopSkipJump
import numpy as np
from utils import apply_augment, create_rotates, create_translates, softmax

tf.disable_eager_execution()


def dists(model_, ds, attack="CW", max_samples=100, input_dim=[None, 32, 32, 3], n_classes=10):
  """Calculate untargeted distance to decision boundary for Adv-x MI attack.

  :param model: model to approximate distances on (attack).
  :param ds: tf dataset should be either the training set or the test set.
  :param attack: "CW" for carlini wagner or "HSJ" for hop skip jump
  :param max_samples: maximum number of samples to take from the ds
  :return: an array of the first samples from the ds, of len max_samples, with the untargeted distances.
  """
  # switch to TF1 style
  sess = K.get_session()
  x = tf.placeholder(dtype=tf.float32, shape=input_dim)
  y = tf.placeholder(dtype=tf.int32, shape=[None, n_classes])
  output = model_(x)
  model = CallableModelWrapper(lambda x: model_(x), "logits")

  if attack == "CW":
    attack = CarliniWagnerL2(model, sess)
    x_adv = attack.generate(x, y=y)
  elif attack == "HSJ":
    attack = HopSkipJumpAttack(model, sess)
    x_adv = attack.generate(x, verbose=False)
  else:
    raise ValueError("Unknown attack {}".format(attack))

  next_element = ds.make_one_shot_iterator().get_next()

  acc = []
  acc_adv = []
  dist_adv = []
  num_samples = 0
  while(True):
    try:
        xbatch, ybatch = sess.run(next_element)
        ybatch = tf.keras.utils.to_categorical(ybatch, n_classes)
        y_pred = sess.run(output, feed_dict={x: xbatch})
        correct = np.argmax(y_pred, axis=-1) == np.argmax(ybatch, axis=-1)
        acc.extend(correct)

        x_adv_np = []
        for i in range(len(xbatch)):
            if correct[i]:
                x_adv_curr = sess.run(x_adv, feed_dict={x: xbatch[i:i+1], y: ybatch[i:i+1]})
            else:
                x_adv_curr = xbatch[i:i+1]
            x_adv_np.append(x_adv_curr)
        x_adv_np = np.concatenate(x_adv_np, axis=0)

        y_pred_adv = sess.run(output, feed_dict={x: x_adv_np})
        corr_adv = np.argmax(y_pred_adv, axis=-1) == np.argmax(ybatch, axis=-1)
        acc_adv.extend(corr_adv)

        d = np.sqrt(np.sum(np.square(x_adv_np-xbatch), axis=(1,2,3)))
        dist_adv.extend(d)
        # print(list(d))
        num_samples += len(xbatch)
        print("processed {} examples".format(num_samples))
        if num_samples >= max_samples:
            break


    except tf.errors.OutOfRangeError as e:
        break

  return dist_adv[:max_samples]


def distance_augmentation_attack(model, train_set, test_set, max_samples, attack_type='d', distance_attack='CW', augment_kwarg=1, batch=100, input_dim=[None, 32, 32, 3], n_classes=10):
  """Combined MI attack using the distanes for each augmentation.

  Args:
    model: model to approximate distances on (attack).
    train_set: the training set for the model
    test_set: the test set for the model
    max_samples: max number of samples to attack
    attack_type: either 'd' or 'r' for translation and rotation attacks, respectively.
    augment_kwarg: the kwarg for each augmentation. If rotations, augment_kwarg defines the max rotation, with n=2r+1
    rotated images being used. If translations, then 4n+1 translations will be used at a max displacement of
    augment_kwarg
    batch: batch size for querying model in the attack.

  Returns: 2D array where rows correspond to samples and columns correspond to the distance to boundary in an untargeted
  attack for that rotated/translated sample.

  """
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
    attack_in[:, i] = dists(model, train_ds, attack=distance_attack, max_samples=max_samples, input_dim=input_dim, n_classes=n_classes)
    attack_out[:, i] = dists(model, test_ds, attack=distance_attack, max_samples=max_samples, input_dim=input_dim, n_classes=n_classes)
  attack_set = (np.concatenate([attack_in, attack_out], 0),
                np.concatenate([train_set[1], test_set[1]], 0),
                m)
  return attack_set


def binary_rand_robust(model, ds, p, max_samples=100, noise_samples=10000, stddev=0.025, input_dim=[None, 107],
                num=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, 250, 500, 1000, 2500, 5000, 10000],
                dataset='adult'):
  """Calculate robustness to random noise for Adv-x MI attack on binary-featureed datasets (+ UCI adult).

    :param model: model to approximate distances on (attack).
    :param ds: tf dataset should be either the training set or the test set.
    :param p: probability for Bernoulli flips for binary features.
    :param max_samples: maximum number of samples to take from the ds
    :param noise_samples: number of noised samples to take for each sample in the ds.
    :param stddev: the standard deviation to use for Gaussian noise (only for Adult, which has some continuous features)
    :param input_dim: dimension of inputs for the dataset.
    :param num: subnumber of samples to evaluate. max number is noise_samples
    :return: a list of lists. each sublist of the accuracy on up to $num noise_samples.
    """
  # switch to TF1 style
  sess = K.get_session()
  x = tf.placeholder(dtype=tf.float32, shape=input_dim)
  output = tf.argmax(model(x), axis=-1)
  next_element = ds.make_one_shot_iterator().get_next()

  num_samples = 0
  robust_accs = [[] for _ in num]
  all_correct = []
  while (True):
    try:
      xbatch, ybatch = sess.run(next_element)
      labels = np.argmax(ybatch, axis=-1)
      y_pred = sess.run(output, feed_dict={x: xbatch})
      correct = y_pred == labels
      all_correct.extend(correct)
      for i in range(len(xbatch)):
        if correct[i]:
          if dataset == 'adult':
            noise = np.random.binomial(1, p, [noise_samples, xbatch[i: i+1, 6:].shape[-1]])
            x_sampled = np.tile(np.copy(xbatch[i:i+1]), (noise_samples, 1))
            x_noisy = np.invert(xbatch[i: i+1, 6:].astype(np.bool), out=np.copy(x_sampled[:, 6:]), where=noise.astype(np.bool)).astype(np.int32)
            noise = stddev * np.random.randn(noise_samples, xbatch[i: i+1, :6].shape[-1])
            x_noisy = np.concatenate([x_sampled[:, :6] + noise, x_noisy], axis=1)
          else:
            noise = np.random.binomial(1, p, [noise_samples, xbatch[i: i + 1].shape[-1]])
            x_sampled = np.tile(np.copy(xbatch[i:i + 1]), (noise_samples, 1))
            x_noisy = np.invert(xbatch[i: i + 1].astype(np.bool), out=x_sampled,
                                where=noise.astype(np.bool)).astype(np.int32)
          preds = []

          bsize = 100
          num_batches = noise_samples // bsize
          for j in range(num_batches):
            preds.extend(sess.run(output, feed_dict={x: x_noisy[j * bsize:(j + 1) * bsize]}))

          for idx, n in enumerate(num):
            if n == 0:
              robust_accs[idx].append(1)
            else:
              robust_accs[idx].append(np.mean(preds[:n] == labels[i]))
        else:
          for idx in range(len(num)):
            robust_accs[idx].append(0)

      num_samples += len(xbatch)
      if num_samples >= max_samples:
        break

    except tf.errors.OutOfRangeError:
      break

  return robust_accs


def continuous_rand_robust(model, ds, max_samples=100, noise_samples=2500, stddev=0.025, input_dim=[None, 32, 32, 3],
                           num=[1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 350, 500, 750, 1000, 1500, 2000, 2500]):
  """Calculate robustness to random noise for Adv-x MI attack on continuous-featureed datasets (+ UCI adult).

  :param model: model to approximate distances on (attack).
  :param ds: tf dataset should be either the training set or the test set.
  :param max_samples: maximum number of samples to take from the ds
  :param noise_samples: number of noised samples to take for each sample in the ds.
  :param stddev: the standard deviation to use for Gaussian noise (only for Adult, which has some continuous features)
  :param input_dim: dimension of inputs for the dataset.
  :param num: subnumber of samples to evaluate. max number is noise_samples
  :return: a list of lists. each sublist of the accuracy on up to $num noise_samples.
  """
  # switch to TF1 style
  sess = K.get_session()
  x = tf.placeholder(dtype=tf.float32, shape=input_dim)
  output = tf.argmax(model(x), axis=-1)
  next_element = ds.make_one_shot_iterator().get_next()

  num_samples = 0
  robust_accs = [[] for _ in num]
  all_correct = []
  while (True):
    try:
      xbatch, ybatch = sess.run(next_element)
      labels = np.argmax(ybatch, axis=-1)
      y_pred = sess.run(output, feed_dict={x: xbatch})
      correct = y_pred == labels
      all_correct.extend(correct)

      x_adv_np = []
      for i in range(len(xbatch)):
        if correct[i]:
          noise = stddev * np.random.randn(noise_samples, input_dim[1:])
          x_noisy = np.clip(xbatch[i:i + 1] + noise, 0, 1)
          preds = []

          bsize = 50
          num_batches = noise_samples // bsize
          for j in range(num_batches):
            preds.extend(sess.run(output, feed_dict={x: x_noisy[j * bsize:(j + 1) * bsize]}))

          for idx, n in enumerate(num):
            if n == 0:
              robust_accs[idx].append(1)
            else:
              robust_accs[idx].append(np.mean(preds[:n] == labels[i]))
        else:
          for idx in range(len(num)):
            robust_accs[idx].append(0)

      num_samples += len(xbatch)
      # print("processed {} examples".format(num_samples))
      # print(correct)
      # print(robust_accs[-len(xbatch):])
      if num_samples >= max_samples:
        break


    # not sure how to iterate over a TF Dataset in TF1.
    # this is ugly but it works
    except tf.errors.OutOfRangeError:
      break

  return robust_accs
