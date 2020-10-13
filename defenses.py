import numpy as np


def memguard(scores):
  """ Given confidence vectors, perform memguard post processing to protect from membership inference.

  Note that this defense assumes the strongest defender that can make arbitrary changes to the confidence vector
  so long as it does not change the label. We as well have the (weaker) constrained optimization that will be
  released at a future data.

  Args:
    scores: confidence vectors as 2d numpy array

  Returns: 2d scores protected by memguard.

  """
  n_classes = scores.shape[1]
  epsilon = 1e-3
  on_score = (1. / n_classes) + epsilon
  off_score = (1. / n_classes) - (epsilon / (n_classes - 1))
  predicted_labels = np.argmax(scores, axis=-1)
  defended_scores = np.ones_like(scores) * off_score
  defended_scores[np.arange(len(defended_scores)), predicted_labels] = on_score
  return defended_scores
