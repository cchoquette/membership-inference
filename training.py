import tensorflow as tf
import argparse
import models
import utils
import numpy as np
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.eager import backprop
from tensorflow.python.distribute import parameter_server_strategy
from types import MethodType
from tensorflow_privacy.privacy.optimizers.dp_optimizer import DPAdamGaussianOptimizer
from tensorflow_privacy.privacy.analysis.gdp_accountant import compute_eps_poisson

# -------------------------------------------------------------------------------------------------------------------
# Script input arguments ------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser('Train and save a model (potentially with a defense')
parser.add_argument('--ndata', type=int, default=2500, help='number of data points to use')
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset to use')
parser.add_argument('--model', type=str, default='make_conv', help='model (fn) to train')
parser.add_argument('--model_depth', type=int, default=1, help='depth of model to use.')
parser.add_argument('--defense', default='', type=str, help='defense to use, including if regularization')
parser.add_argument('--target_weight_path', default='', type=str, help='path to target weights for transfer learning')
parser.add_argument('--source_weight_path', default='', type=str, help='path to source weights for transfer learning')
parser.add_argument('--target_model_path', default='target_model.tf', type=str, help='path to save final model.')
parser.add_argument('--source_model_path', default='source_model.tf', type=str, help='path to save fianl model')
parser.add_argument('--reg_constant', default=1., type=float, help='regularization constant to use, if any, eps for DP,'
                                                                   'dropout rate for dropout, l1/l2 lambda, ')
parser.add_argument('--batch_size', default=100, type=int, help='batch size for training target/source models.')
parser.add_argument('--noise', default=0.5, type=float, help='noise ratio for DP training')
args = parser.parse_args()

valid_regs = ['l2', 'l1', 'dropout', 'dp', 'last_layer',
              'full_fine_tune', 'dp_last_layer', 'dp_full_fine_tune', '']

if args.defense not in valid_regs:
  if args.defense == 'adversarial_regularization':
    print('adversarial regularization should be trained separately using adv_reg_training.py')
  raise ValueError(f'{args.defense} is not a valid defense. Valids are: {valid_regs}')

# -------------------------------------------------------------------------------------------------------------------
# Functions and classes start ------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

def train_step(self, data):
  """The logic for one training step.
  This method can be overridden to support custom training logic.
  This method is called by `Model.make_train_function`.
  This method should contain the mathemetical logic for one step of training.
  This typically includes the forward pass, loss calculation, backpropagation,
  and metric updates.
  Configuration details for *how* this logic is run (e.g. `tf.function` and
  `tf.distribute.Strategy` settings), should be left to
  `Model.make_train_function`, which can also be overridden.
  Arguments:
    data: A nested structure of `Tensor`s.
  Returns:
    A `dict` containing values that will be passed to
    `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
    values of the `Model`'s metrics are returned. Example:
    `{'loss': 0.2, 'accuracy': 0.7}`.
  """
  # These are the only transformations `Model.fit` applies to user-input
  # data when a `tf.data.Dataset` is provided. These utilities will be exposed
  # publicly.
  data = data_adapter.expand_1d(data)
  x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

  with backprop.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y_pred = self(x, training=True)
    loss = self.compiled_loss(
      y, y_pred, sample_weight, regularization_losses=self.losses)
    if self.optimizer._num_microbatches is None:
      self.optimizer._num_microbatches = tf.shape(input=loss)[0]
    losses = [tf.reduce_mean(input_tensor=tf.gather(loss, [idx])) for idx in range(self.optimizer._num_microbatches)]
  final_grads = [tape.gradient(losses[i], self.trainable_variables) for i in range(self.optimizer._num_microbatches)]
  sample_params = (
    self.optimizer._dp_sum_query.derive_sample_params(self.optimizer._global_state))

  var_list = self.trainable_variables
  sample_state = self.optimizer._dp_sum_query.initial_sample_state(var_list)
  for grads in final_grads:
    grads_list = [g if g is not None else tf.zeros_like(v) for (g, v) in zip(list(grads), var_list)]
    sample_state = self.optimizer._dp_sum_query.accumulate_record(
      sample_params, sample_state, grads_list)

  grad_sums, self.optimizer._global_state = (
    self.optimizer._dp_sum_query.get_noised_result(
      sample_state, self.optimizer._global_state))

  def normalize(v):
    return tf.truediv(v, tf.cast(self.optimizer._num_microbatches, tf.float32))

  final_grads = tf.nest.map_structure(normalize, grad_sums)

  self.optimizer._was_compute_gradients_called = True
  self.optimizer.apply_gradients(zip(final_grads, self.trainable_variables))
  self.compiled_metrics.update_state(y, y_pred, sample_weight)
  del tape  # since persistent, we need to garbage collect.
  return {m.name: m.result() for m in self.metrics}


def train_model(models, train_set, **kwargs):
  epochs = []
  losses = []
  for i in range(len(models)):
    train_lbl_sel = np.argmax(train_set[1], axis=1) == i
    if np.sum(train_lbl_sel.astype(np.int)) <= 0.:
      continue
    history = models[i].fit(train_set[0][train_lbl_sel],
                            train_set[2][train_lbl_sel],
                            batch_size=kwargs.get('batch_size', 1),
                            callbacks=[tf.keras.callbacks.EarlyStopping(
                              monitor='val_loss', mode='min', patience=20)],
                            # validation_data=(val_set[0][val_lbl_sel],
                            #                  val_set[2][val_lbl_sel]),
                            epochs=kwargs.get('epochs', 1),
                            shuffle=True, verbose=0,
                            )
    hist = history.history
    keys = hist.keys()
    hist = {key: hist[key][-1] for key in keys if key == 'loss' or key == 'sparse_categorical_accuracy'}
    epochs.append(len(history.history['loss']))
    losses.append(hist['loss'])
    # print(f"label: '{i}' has {hist}")
  return models, np.mean(losses)


class EpsilonTracker(tf.keras.callbacks.Callback):
  def __init__(self, noise_multiplier, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.eps = []
    self.noise_multiplier = noise_multiplier

  def on_epoch_end(self, epoch, logs={}):
    # eps, _ = compute_eps_poisson(flags_obj.class_size*data.N_CLASSES,
    #                              flags_obj.train_batch_size,
    #                              self.noise_multiplier, epoch,
    #                              (1./flags_obj.class_size*data.N_CLASSES)/10)
    eps = compute_eps_poisson(epoch, self.noise_multiplier,
                              args.ndata,
                              args.batch_size,
                              1. / args.ndata)
    logs['eps'] = eps
    # if epoch % 10 == 0:
    #   print(f"current eps value is: {eps}")
    self.eps.append(eps)


class EarlyStopping(tf.keras.callbacks.Callback):
  """Stop training when a monitored metric has stopped improving.
  Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be 'loss', and mode would be 'min'. A
  `model.fit()` training loop will check at end of every epoch whether
  the loss is no longer decreasing, considering the `min_delta` and
  `patience` if applicable. Once it's found no longer decreasing,
  `model.stop_training` is marked True and the training terminates.
  The quantity to be monitored needs to be available in `logs` dict.
  To make it so, pass the loss or metrics at `model.compile()`.
  Example:
  >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  >>> # This callback will stop the training when there is no improvement in
  >>> # the validation loss for three consecutive epochs.
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=10, batch_size=1, callbacks=[callback],
  ...                     verbose=0)
  >>> len(history.history['loss'])  # Only 4 epochs are run.
  4
  """

  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               restore_best_weights=False):
    """Initialize an EarlyStopping callback.
    Arguments:
        monitor: Quantity to be monitored.
        min_delta: Minimum change in the monitored quantity
            to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no
            improvement.
        patience: Number of epochs with no improvement
            after which training will be stopped.
        verbose: verbosity mode.
        mode: One of `{"auto", "min", "max"}`. In `min` mode,
            training will stop when the quantity
            monitored has stopped decreasing; in `max`
            mode it will stop when the quantity
            monitored has stopped increasing; in `auto`
            mode, the direction is automatically inferred
            from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
        restore_best_weights: Whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """
    super(EarlyStopping, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.best_weights = None

    if mode not in ['auto', 'min', 'max']:
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.wait = 0
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
    else:
      print(f"stopping eps of: {current}")
      self.wait += 1
      if self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        if self.restore_best_weights:
          if self.verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    return monitor_value

# -------------------------------------------------------------------------------------------------------------------
# Functions and classes end, training script start ------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------

model = getattr(models, args.model)  # get model_fn

# get datasets for target / source models
target_train_set, target_test_set, source_train_set, source_test_set, input_dim, n_classes = utils.get_data(args.dataset, args.ndata)
regularization = "none"
if args.defense in ['l1', 'l2', 'dropout']:
  regularization == args.regularization
elif args.defense in ['', 'dp', 'advreg', 'fine-tune', 'whole']:  # last two are transfer learning
  regularization = 'none'  # these train the model differently, which we will do at training time below.
else:
  raise ValueError(f"Defense: {args.defense} not valid")


# define and compile model
target_model = model(input_dim, args.model_depth, regularization, args.reg_constant, n_classes)
source_model = model(input_dim, args.model_depth, regularization, args.reg_constant, n_classes)
t_optim = tf.keras.optimizers.Adam()
t_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
t_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
s_optim = tf.keras.optimizers.Adam()
s_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
s_metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
t_cbs = []
s_cbs = []
if args.defense == 'dp':
  target_model.train_step = MethodType(train_step, target_model)
  source_model.train_step = MethodType(train_step, source_model)
  t_optim = DPAdamGaussianOptimizer(learning_rate=0.0001,
                                  num_microbatches=args.batch_size,
                                  noise_multiplier=args.noise,
                                  l2_norm_clip=2.)
  s_optim = DPAdamGaussianOptimizer(learning_rate=0.0001,
                                    num_microbatches=args.batch_size,
                                    noise_multiplier=args.noise,
                                    l2_norm_clip=2.)
  t_cbs.extend([EpsilonTracker(args.noise),
   EarlyStopping(mode='min', patience=0,
                 min_delta=0,
                 monitor='eps',
                 baseline=args.reg_constant),
   ])
  s_cbs.extend([EpsilonTracker(args.noise),
                EarlyStopping(mode='min', patience=0,
                              min_delta=0,
                              monitor='eps',
                              baseline=args.reg_constant),
                ])

elif args.defense in ['full_fine_tune', 'last_layer', 'dp_full_fine_tune', 'dp_last_layer']:
  if args.defense in ['last_layer', 'dp_last_layer']:
    ll = target_model.layers[-1]
    target_model.pop()
  target_model.load_weights(args.target_weight_path)
  if args.defense in ['last_layer', 'dp_last_layer']:
    model.trainable = False
    ll.trainable = True
  if args.defense in ['dp_full_fine_tune', 'dp_last_layer']:
    target_model.train_step = MethodType(train_step, target_model)
    source_model.train_step = MethodType(train_step, source_model)
    t_optim = DPAdamGaussianOptimizer(learning_rate=0.0001,
                                    num_microbatches=args.batch_size,
                                    noise_multiplier=args.noise,
                                    l2_norm_clip=2.)
    s_optim = DPAdamGaussianOptimizer(learning_rate=0.0001,
                                      num_microbatches=args.batch_size,
                                      noise_multiplier=args.noise,
                                      l2_norm_clip=2.)
    t_cbs.extend([EpsilonTracker(args.noise),
                EarlyStopping(mode='min', patience=0,
                              min_delta=0,
                              monitor='eps',
                              baseline=args.reg_constant),
                ])
    s_cbs.extend([EpsilonTracker(args.noise),
                  EarlyStopping(mode='min', patience=0,
                                min_delta=0,
                                monitor='eps',
                                baseline=args.reg_constant),
                  ])
else:
  t_cbs.append(tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.5e-2, patience=20))
  s_cbs.append(tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.5e-2, patience=20))

target_model.compile(optimizer=t_optim, loss=t_loss, metrics=t_metrics)
source_model.compile(optimizer=s_optim, loss=s_loss, metrics=s_metrics)

target_model.fit(*target_train_set, epochs=1000, callbacks=t_cbs, batch_size=args.batch_size, verbose=2)
source_model.fit(*source_train_set, epochs=1000, callbacks=s_cbs, batch_size=args.batch_size, verbose=2)

target_model.save(args.target_model_path)
source_model.save(args.source_model_path)
