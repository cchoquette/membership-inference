import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf  # works with TF-GPU 1.15.0
from tensorflow.python.keras import backend as K

# SILENCE!
import logging
import warnings
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd
from collections import defaultdict
from attacks import get_threshold, train_best_attack_model
from utils import get_data, softmax
import attack_features, attack_features_tf1
from defenses import memguard


import argparse

parser = argparse.ArgumentParser('Take a trained model and attack it')
parser.add_argument('--ndata', default=2500, type=int, help='total amount of data to use for training/test')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset name to attack')
parser.add_argument('--target_model_path', default='target_model.tf', type=str, help='path to load final model.')
parser.add_argument('--source_model_path', default='source_model.tf', type=str, help='path to load fianl model')
parser.add_argument('--attack_batch_size', default=10, type=int, help='batch size for predictions in attack')
parser.add_argument('--defense', default='', type=str, help='name of post processing defense to use.')
parser.add_argument('--attacks', default='bndac', type=str, help='each char is an attack to perform. n is confidence-vector.'
                                                                'd is translation, r is rotation, a is adv-e (boundary distance, label-only),'
                                                                'b is for baseline gap attack'
                                                                'g is gaussian noise, w is white-box CW, '
                                                                'c is combined (translation + distance)')
parser.add_argument('--r', default=9, type=int, help='r param in rotation attack if used')
parser.add_argument('--d', default=1, type=int, help='d param in translation attack if used')
parser.add_argument('--noise_samples', default=5000, type=int, help='number of times to duplicate and noise each sample.')
parser.add_argument('--max_samples', default=-1, type=int, help='-1 to use ndata, else the max number of samples to process for longer attacks (dist, combined)')
args = parser.parse_args()


valid_defenses = ['memguard']
if args.defense not in valid_defenses:
  raise ValueError(f"Defense: {args.defense} is not a valid defense. Valid defenses are: {valid_defenses}")


def attack(args):
  batch = args.attack_batch_size
  target_train_set, target_test_set, source_train_set, source_test_set, input_dim, n_classes = get_data(args.dataset, args.ndata)
  source_labels = np.concatenate([source_train_set[1].flatten(), source_test_set[1].flatten()],
                                 axis=0)
  target_labels = np.concatenate([target_train_set[1].flatten(), target_test_set[1].flatten()], axis=0)
  print(f"source_labels: {source_labels.shape}, target_labels: {target_labels.shape}")
  target_model = tf.keras.models.load_model(args.target_model_path)
  source_model = tf.keras.models.load_model(args.source_model_path)

  target_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       optimizer=tf.keras.optimizers.Adam(),
                       metrics=['accuracy'])
  source_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       optimizer=tf.keras.optimizers.Adam(),
                       metrics=['accuracy'])
  _, bstrainacc = source_model.evaluate(*source_train_set, verbose=2)
  _, bstestacc = source_model.evaluate(*source_test_set, verbose=2)

  _, bttrainacc = target_model.evaluate(*target_train_set, verbose=2)
  _, bttestacc = target_model.evaluate(*target_test_set, verbose=2)
  # get softmax features for boundary distance and prediction vector attacsk (n, a)
  target_train_ds = tf.data.Dataset.from_tensor_slices(target_train_set).batch(batch)
  target_test_ds = tf.data.Dataset.from_tensor_slices(target_test_set).batch(batch)
  source_train_ds = tf.data.Dataset.from_tensor_slices(source_train_set).batch(batch)
  source_test_ds = tf.data.Dataset.from_tensor_slices(source_test_set).batch(batch)

  source_in = source_model.predict(source_train_ds)
  source_out = source_model.predict(source_test_ds)

  target_in = target_model.predict(target_train_ds)
  target_out = target_model.predict(target_test_ds)

  source_in = softmax(source_in)
  source_out = softmax(source_out)
  target_in = softmax(target_in)
  target_out = softmax(target_out)

  if args.defense == 'memguard':
    target_in = memguard(target_in)
    target_out = memguard(target_out)

  for attack in list(args.attacks):

    target_features = np.concatenate([target_in, target_out], axis=0)
    source_features = np.concatenate([source_in, source_out], axis=0)
    print(f"source_features: {source_features.shape}, target_features: {target_features.shape}")
    source_m = np.concatenate([np.ones(len(source_in)),
                               np.zeros(len(source_out))], axis=0)
    target_m = np.concatenate([np.ones(len(target_in)),
                               np.zeros(len(target_out))], axis=0)

    # downsample the dataset since the attack is so slow.
    # make sure the dataset is shuffled so that we don't attack the same class all the time!
    if attack == 'b':
      print(f"Gap attack| source: {50 + (bstrainacc - bstestacc) * 50}, target: {50 + (bttrainacc - bttestacc) * 50}")
    elif attack == 'a':
      max_samples = args.ndata if args.max_samples == -1 else args.max_samples
      source_m = np.concatenate([np.ones(max_samples),
                               np.zeros(max_samples)], axis=0)
      target_m = np.concatenate([np.ones(max_samples),
                                 np.zeros(max_samples)], axis=0)
      # attack with HipSkipJump (very slow)
      dists_source_in = attack_features_tf1.dists(source_model, source_train_ds, attack="HSJ", max_samples=max_samples,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      dists_source_out = attack_features_tf1.dists(source_model, source_test_ds, attack="HSJ", max_samples=max_samples,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      dists_source = np.concatenate([dists_source_in, dists_source_out], axis=0)
      dists_target_in = attack_features_tf1.dists(target_model, target_train_ds, attack="HSJ", max_samples=max_samples,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      dists_target_out = attack_features_tf1.dists(target_model, target_test_ds, attack="HSJ", max_samples=max_samples,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      dists_target = np.concatenate([dists_target_in, dists_target_out], axis=0)
      print("threshold on HSJ:")
      acc2, prec2, _, _ = get_threshold(source_m, dists_source, target_m, dists_target)
    elif attack == 'w':
      max_samples = args.ndata if args.max_samples == -1 else args.max_samples
      source_m = np.concatenate([np.ones(max_samples),
                                 np.zeros(max_samples)], axis=0)
      target_m = np.concatenate([np.ones(max_samples),
                                 np.zeros(max_samples)], axis=0)
      # attack with C&W
      dists_source_in = attack_features_tf1.dists(source_model, source_train_ds, attack="CW", max_samples=max_samples,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      dists_source_out = attack_features_tf1.dists(source_model, source_test_ds, attack="CW", max_samples=max_samples,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      dists_source = np.concatenate([dists_source_in, dists_source_out], axis=0)
      dists_target_in = attack_features_tf1.dists(target_model, target_train_ds, attack="CW", max_samples=max_samples,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      dists_target_out = attack_features_tf1.dists(target_model, target_test_ds, attack="CW", max_samples=max_samples,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      dists_target = np.concatenate([dists_target_in, dists_target_out], axis=0)
      print("threshold on C&W:")
      acc1, prec1, _, _ = get_threshold(source_m, dists_source, target_m, dists_target)

    elif attack == 'n':
      # just look at confidence in predicted label
      conf_source = np.max(source_features, axis=-1)
      conf_target = np.max(target_features, axis=-1)
      print("threshold on predicted label:")
      acc1, prec1, _, _ = get_threshold(source_m, conf_source, target_m, conf_target)

      # look at confidence in true label
      conf_source = source_features[range(len(source_features)), source_labels]
      conf_target = target_features[range(len(target_features)), target_labels]
      print("threshold on true label:")
      acc2, prec2, _, _ = get_threshold(source_m, conf_source, target_m, conf_target)
    elif attack == 'c':
      max_samples = args.ndata if args.max_samples == -1 else args.max_samples
      aug_kwarg = args.d if attack == 'd' else args.r
      attack_test_set = attack_features_tf1.distance_augmentation_attack(target_model, target_train_set, target_test_set,
                                                            max_samples,
                                                            attack,
                                                            aug_kwarg, args.attack_batch_size,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      attack_train_set = attack_features_tf1.distance_augmentation_attack(source_model, source_train_set, source_test_set,
                                                             max_samples,
                                                             attack,
                                                             aug_kwarg, args.attack_batch_size,
                                                  input_dim=[None, input_dim[0], input_dim[1], input_dim[2]],
                                                  n_classes=n_classes)
      vals = train_best_attack_model(attack_train_set, attack_test_set, attack, n_classes=n_classes)
    elif attack == 'd' or attack == 'r':
      max_samples = args.ndata
      aug_kwarg = args.d if attack == 'd' else args.r
      attack_test_set = attack_features.augmentation_attack(target_model, target_train_set, target_test_set, max_samples,
                                                            attack,
                                                       aug_kwarg, args.attack_batch_size)
      attack_train_set = attack_features.augmentation_attack(source_model, source_train_set, source_test_set, max_samples,
                                                            attack,
                                                            aug_kwarg, args.attack_batch_size)
      vals = train_best_attack_model(attack_train_set, attack_test_set, attack, n_classes=n_classes)
    elif attack == 'g':
      max_samples = min(10000, args.ndata)
      if args.dataset in ['adult', 'purchase', 'texas', 'location']:
        sigmas = [1. / input_dim, 2. / input_dim, 3. / input_dim, 5. / input_dim, 10. / input_dim]
        if args.dataset == 'adult':
          sigmas = [20. / input_dim, 30. / input_dim, 50. / input_dim]
        for sigma in sigmas:
          print(f"threshold on noise robustness, sigma: {sigma}")
          noise_source_in = attack_features_tf1.binary_rand_robust(source_model, source_train_ds, stddev=sigma * input_dim,
                                                                   p=sigma,
                                                                       max_samples=max_samples,
                                                                       input_dim=[None, input_dim],
                                                                       noise_samples=args.noise_samples,
                                                                   dataset=args.dataset)
          noise_source_out = attack_features_tf1.binary_rand_robust(source_model, source_test_ds, stddev=sigma * input_dim,
                                                                    p=sigma,
                                                                        max_samples=max_samples,
                                                                        input_dim=[None, input_dim],
                                                                        noise_samples=args.noise_samples,
                                                                    dataset=args.dataset)
          noise_target_in = attack_features_tf1.binary_rand_robust(target_model, target_train_ds, stddev=sigma * input_dim,
                                                                   p=sigma,
                                                                       max_samples=max_samples,
                                                                       input_dim=[None, input_dim],
                                                                       noise_samples=args.noise_samples, dataset=args.dataset)
          noise_target_out = attack_features_tf1.binary_rand_robust(target_model, target_test_ds, stddev=sigma * input_dim,
                                                                    p=sigma,
                                                                        max_samples=max_samples,
                                                                        input_dim=[None, input_dim],
                                                                        noise_samples=args.noise_samples, dataset=args.dataset)

          for i in range(len(noise_source_in)):
            noise_source = np.concatenate([noise_source_in[i], noise_source_out[i]], axis=0)
            noise_target = np.concatenate([noise_target_in[i], noise_target_out[i]], axis=0)
            get_threshold(source_m, noise_source, target_m, noise_target)
      else:
        for sigma in [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
          print(f"threshold on noise robustness, sigma: {sigma}")
          noise_source_in = attack_features_tf1.continuous_rand_robust(source_model, source_train_ds, stddev=sigma, max_samples=max_samples, input_dim=[None, input_dim], noise_samples=args.noise_samples)
          noise_source_out = attack_features_tf1.continuous_rand_robust(source_model, source_test_ds, stddev=sigma, max_samples=max_samples, input_dim=[None, input_dim], noise_samples=args.noise_samples)
          noise_target_in = attack_features_tf1.continuous_rand_robust(target_model, target_train_ds, stddev=sigma, max_samples=max_samples, input_dim=[None, input_dim], noise_samples=args.noise_samples)
          noise_target_out = attack_features_tf1.continuous_rand_robust(target_model, target_test_ds, stddev=sigma, max_samples=max_samples, input_dim=[None, input_dim], noise_samples=args.noise_samples)

          for i in range(len(noise_source_in)):
            noise_source = np.concatenate([noise_source_in[i], noise_source_out[i]], axis=0)
            noise_target = np.concatenate([noise_target_in[i], noise_target_out[i]], axis=0)
            get_threshold(source_m, noise_source, target_m, noise_target)
    else:
      raise ValueError(f'attack: {attack} not supported.')

attack(args)