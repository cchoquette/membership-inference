print("adv_reg_training")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf  # works with TF-GPU 1.15.0
print(tf.__version__)
print(tf.executing_eagerly())

# SILENCE!
import logging
import warnings
tf.get_logger().setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('ndata', type=int)
parser.add_argument('k', type=int)
parser.add_argument('l', type=float)
parser.add_argument('seed', type=int)
args = parser.parse_args()

print(args.ndata, args.k, args.l, args.seed)
np.random.seed(args.seed)

def softmax(x):
  if len(np.shape(x)) == 1:
    x = x[np.newaxis, :]
  s = np.max(x, axis=1)[:, np.newaxis]
  e_x = np.exp(x - s)
  div = np.sum(e_x, axis=1)
  div = div[:, np.newaxis]
  return e_x / div


def make_conv(input_shape, depth, regularization, reg_constant, num_classes, k):
  """Return conv_model as tf.keras.Model, without Softmax layer (only logits).

  To perform

  :param input_shape:
  :param depth:
  :param regularization:
  :param reg_constant:
  :param num_classes:
  :return:
  """
  if len(input_shape) == 2:
    data_format = 'channels_first'
  elif len(input_shape) == 3 and (input_shape[0] == 1 or input_shape[0] == 3):
    data_format = 'channels_first'
  else:
    data_format = 'channels_last'
  conv_model = tf.keras.models.Sequential()
  conv_model.add(tf.keras.Input(input_shape))
  # conv_model = EpochSequential()
  for _ in range(depth):
    conv_model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                          data_format=data_format))
    conv_model.add(tf.keras.layers.Activation('relu'))
    conv_model.add(tf.keras.layers.Conv2D(32, (3, 3), data_format=data_format))
    conv_model.add(tf.keras.layers.Activation('relu'))
  conv_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  conv_model.add(tf.keras.layers.Dropout(0.25))
  for _ in range(depth):
    conv_model.add(tf.keras.layers.Conv2D(64, (3, 3), padding='same',
                                          data_format=data_format))
    conv_model.add(tf.keras.layers.Activation('relu'))
    conv_model.add(tf.keras.layers.Conv2D(64, (3, 3), data_format=data_format))
    conv_model.add(tf.keras.layers.Activation('relu'))
  conv_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
  if regularization == 'l1':
    k_reg = tf.keras.regularizers.L1L2(l1=reg_constant)
  elif regularization == 'l2':
    k_reg = tf.keras.regularizers.L1L2(l2=reg_constant)
  else:
    k_reg = None
  if regularization == 'dropout':
    conv_model.add(tf.keras.layers.Dropout(reg_constant))
  else:
    conv_model.add(tf.keras.layers.Dropout(0.0))
  conv_model.add(tf.keras.layers.Flatten())
  conv_model.add(tf.keras.layers.Dense(512, kernel_regularizer=k_reg))
  conv_model.add(tf.keras.layers.Activation('relu'))
  if regularization == 'dropout':
    conv_model.add(tf.keras.layers.Dropout(reg_constant))
  else:
    conv_model.add(tf.keras.layers.Dropout(0.0))
  conv_model.add(tf.keras.layers.Dense(num_classes, kernel_regularizer=k_reg, activation='softmax'))
  return conv_model


input_x = tf.keras.Input(shape=(10,))
input_y = tf.keras.Input(shape=(10,))
x = tf.keras.layers.Dense(1024, activation='relu')(input_x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.models.Model(inputs=input_x, outputs=x)
y = tf.keras.layers.Dense(512, activation='relu')(input_y)
y = tf.keras.layers.Dense(64, activation='relu')(y)
y = tf.keras.models.Model(inputs=input_y, outputs=y)
out = tf.keras.layers.concatenate([x.output, y.output], axis=-1)
out = tf.keras.layers.Dense(256, activation='relu')(out)
out = tf.keras.layers.Dense(64, activation='relu')(out)
out = tf.keras.layers.Dense(2, activation='softmax')(out)
advregmodel = tf.keras.models.Model(inputs=[x.input, y.input], outputs=out)


def adv_reg_training(model, advregmodel, train_set, mi_test_set, test_set, l, k, epochs, batch_size):
  total_steps = epochs * k * len(train_set[0]) // batch_size
  steps_per_epoch = len(train_set[0]) // batch_size
  minmaxstep = 1
  adv_losses = []
  priv_losses = []
  classifier_losses = []
  model_losses = []
  print(f"training for: {total_steps} steps with k: {k} and l: {l}")
  for step in range(total_steps):
    if minmaxstep % k == 0 or step < 3 * steps_per_epoch:
      train_indices = np.random.choice(np.arange(len(train_set[1])), batch_size, False)
      with tf.GradientTape() as tape:
        logits = model(train_set[0][train_indices], training=True)
        loss_value = model.loss(train_set[1][train_indices], logits)
        adv_preds = advregmodel([logits, tf.keras.utils.to_categorical(train_set[1][train_indices], num_classes=10)], training=False)
        adv_preds = tf.gather(adv_preds, tf.convert_to_tensor(1, dtype=tf.int32), axis=1)
        privacy_loss = tf.math.log(adv_preds)
        priv_losses.append(tf.reduce_mean(privacy_loss))
        privacy_loss = l * privacy_loss
        # print(advregmodel([softmax(logits), tf.keras.utils.to_categorical(train_set[1][train_indices], num_classes=10)], training=False), + 1e-12)
        model_losses.append(tf.reduce_mean(loss_value))
        loss_value += privacy_loss
        if step % 1000 == 0:
          print(loss_value.shape, privacy_loss.shape, adv_preds.shape)
        loss_value = tf.reduce_mean(loss_value)
      grads = tape.gradient(loss_value, model.trainable_weights)
      model.optimizer.apply_gradients(zip(grads, model.trainable_weights))
      classifier_losses.append(loss_value)
    else:
      # minmaxstep += 1
      # continue
      train_indices = np.random.choice(np.arange(len(train_set[1])), batch_size, False)
      mi_test_indices = np.random.choice(np.arange(len(mi_test_set[1])), batch_size, False)
      in_preds = softmax(model.predict(train_set[0][train_indices]))
      out_preds = softmax(model.predict(mi_test_set[0][mi_test_indices]))
      mi_training = np.concatenate([in_preds, out_preds], 0)
      training_labels = np.concatenate([train_set[1][train_indices], mi_test_set[1][mi_test_indices]], 0)
      training_labels = tf.keras.utils.to_categorical(training_labels, num_classes=10)
      mi_labels = np.concatenate([np.ones(len(in_preds)), np.zeros(len(out_preds))], 0)
      with tf.GradientTape() as tape:
        logits = advregmodel([mi_training, training_labels], training=True)
        loss_value = advregmodel.loss(mi_labels.reshape((-1, 1)), logits)
      grads = tape.gradient(loss_value, advregmodel.trainable_weights)
      advregmodel.optimizer.apply_gradients(zip(grads, advregmodel.trainable_weights))
      adv_losses.append(loss_value)
    max_print = 250
    if step % 500 == 0:
      print(f"adv_loss: {np.mean(adv_losses[-max_print:])}, classifier_loss: {np.mean(classifier_losses[-max_print:])}, model_loss: {np.mean(model_losses[-max_print:])},  privacy_loss: {np.mean(priv_losses[-max_print:])}, step:{step} of {total_steps}")
    minmaxstep += 1
    if step % steps_per_epoch == 0 and np.mean(classifier_losses[-steps_per_epoch*20:]) < np.mean(classifier_losses[-steps_per_epoch:]) and step > steps_per_epoch * 100:
      print(f'breaking early at step: {step} and epoch: {step // steps_per_epoch}')
      break


def convert_data(x, y, ndata, skip):
  x = x.astype(np.float32)
  start = skip * ndata
  end = start + ndata
  x = x[start:end]
  y = y[start:end]
  # y = tf.keras.utils.to_categorical(y, num_classes=10)
  x /= 255.
  p = np.random.permutation(len(x))
  x = x[p]
  y = y[p]
  return x, y


ndata = args.ndata
print(ndata, args.k, args.l)
k = args.k
l = args.l
train_set, test_set = tf.keras.datasets.cifar10.load_data()
all_data = (np.concatenate([train_set[0], test_set[0]], 0),
            np.concatenate([train_set[1], test_set[1]], 0))
target_train_set = convert_data(*all_data, ndata, 0)
target_test_set = convert_data(*all_data, ndata, 1)

source_train_set = convert_data(*all_data, ndata, 2)
source_test_set = convert_data(*all_data, ndata, 3)

mi_target_test_set = convert_data(*all_data, ndata, 4)
mi_source_test_set = convert_data(*all_data, ndata, 5)

init_weights = advregmodel.get_weights()
advregmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
target_model = make_conv((32, 32, 3), 1, 'none', 1, 10, k)
source_model = make_conv((32, 32, 3), 1, 'none', 1, 10, k)
target_model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE),
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
adv_reg_training(target_model, advregmodel, target_train_set, mi_target_test_set, target_test_set, l,
                                  k, 1000, 10)
advregmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
advregmodel.set_weights(init_weights)

source_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE),
                     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
adv_reg_training(source_model, advregmodel, source_train_set, mi_source_test_set, source_test_set, l,
                                  k, 1000, 10)
_, bstrainacc = source_model.evaluate(*source_train_set, verbose=2)
_, bstestacc = source_model.evaluate(*source_test_set, verbose=2)

_, bttrainacc = target_model.evaluate(*target_train_set, verbose=2)
_, bttestacc = target_model.evaluate(*target_test_set, verbose=2)
target_model.save(f'models/saved/target_{args.ndata}_{args.k}_{args.l}_{args.seed}.tf')
source_model.save(f'models/saved/source_{args.ndata}_{args.k}_{args.l}_{args.seed}.tf')
print(bstrainacc, bstestacc, bttestacc, bttrainacc)
advregmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
advregmodel.set_weights(init_weights)
# target_features = np.concatenate([softmax(target_model.predict(target_train_set[0])), softmax(target_model.predict(target_test_set[0]))], axis=0)
target_features = np.concatenate([target_model.predict(target_train_set[0]), target_model.predict(target_test_set[0])], axis=0)
target_features = [target_features, tf.keras.utils.to_categorical(np.concatenate([target_train_set[1], target_test_set[1]], 0), 10)]
mi_labels = np.concatenate([np.ones(len(target_train_set[0])), np.zeros(len(target_test_set[0]))], axis=0).reshape((-1, 1))
print(target_features[0].shape, target_features[1].shape, mi_labels.shape)
advregmodel.fit(target_features, mi_labels,
                shuffle=True, batch_size=10, verbose=2, epochs=100)

print(f"target MI: {advregmodel.evaluate(target_features, mi_labels, verbose=2)}")

advregmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                    metrics=['accuracy'])
advregmodel.set_weights(init_weights)
source_features = np.concatenate([source_model.predict(source_train_set[0]), source_model.predict(source_test_set[0])], axis=0)
source_features = [source_features, tf.keras.utils.to_categorical(np.concatenate([source_train_set[1], source_test_set[1]], 0), 10)]
advregmodel.fit(source_features, mi_labels,
                shuffle=True, batch_size=10, verbose=2, epochs=100)
print(f"source MI: {advregmodel.evaluate(source_features, mi_labels, verbose=2)}")

