import tensorflow as tf


def make_conv(input_shape, depth, regularization, reg_constant, num_classes):
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
  for _ in range(depth):
    conv_model.add(tf.keras.layers.Conv2D(32, (3, 3), padding='same',
                                          data_format=data_format))
    conv_model.add(tf.keras.layers.Activation('relu'))
    conv_model.add(tf.keras.layers.Conv2D(32, (3, 3), data_format=data_format))
    conv_model.add(tf.keras.layers.Activation('relu'))
  conv_model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
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
  conv_model.add(tf.keras.layers.Dense(num_classes, kernel_regularizer=k_reg))
  return conv_model


def make_fc(input_shape, depth, regularization, reg_constant, num_classes):
  """Return conv_model as tf.keras.Model, without Softmax layer (only logits).

  To perform

  :param input_shape:
  :param depth:
  :param regularization:
  :param reg_constant:
  :param num_classes:
  :return:
  """
  fc_model = tf.keras.models.Sequential()
  if regularization == 'l1':
    k_reg = tf.keras.regularizers.L1L2(l1=reg_constant)
  elif regularization == 'l2':
    k_reg = tf.keras.regularizers.L1L2(l2=reg_constant)
  else:
    k_reg = None
  fc_model.add(tf.keras.Input(input_shape))
  # conv_model = EpochSequential()
  if regularization == 'dropout':
    fc_model.add(tf.keras.layers.Dropout(reg_constant))
  else:
    fc_model.add(tf.keras.layers.Dropout(0.0))
  for _ in range(depth):
    fc_model.add(tf.keras.layers.Dense(128, kernel_regularizer=k_reg))
    fc_model.add(tf.keras.layers.Activation('tanh'))
    if regularization == 'dropout':
      fc_model.add(tf.keras.layers.Dropout(reg_constant))
    else:
      fc_model.add(tf.keras.layers.Dropout(0.0))
  fc_model.add(tf.keras.layers.Dense(num_classes, kernel_regularizer=k_reg))
  return fc_model
