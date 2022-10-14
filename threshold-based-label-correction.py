import numpy as np
import pandas as pd

original_train_labels = pd.read_csv("./data/training-set.csv")["is_suicide"]
original_test_labels = pd.read_csv("./data/testing-set.csv")["is_suicide"]

original_train_labels = np.asarray(original_train_labels)
original_test_labels = np.asarray(original_test_labels)

predicted_labels = pd.read_csv("gmm_combined_prob.csv", header=None)[1]
predicted_probs = pd.read_csv("gmm_combined_prob.csv", header=None)[0]

predicted_labels = np.asarray(predicted_labels)
predicted_probs = np.asarray(predicted_probs)

predicted_train_labels = predicted_labels[0:len(original_train_labels)]
predicted_train_probs = predicted_probs[:len(original_train_labels)]

predicted_test_labels = predicted_labels[len(original_train_labels):]
predicted_test_probs = predicted_probs[len(original_train_labels):]

tau = 0.90

final_train_labels = []

for i in range(len(original_train_labels)):
  if original_train_labels[i] != predicted_train_labels[i]:
    if predicted_train_probs[i] > tau or predicted_train_probs[i] < (1 - tau):
      final_train_labels.append(predicted_train_labels[i])
    else:
      final_train_labels.append(original_train_labels[i])
  else:
    final_train_labels.append(original_train_labels[i])

final_train_labels = np.asarray(final_train_labels)

final_test_labels = []

for i in range(len(original_test_labels)):
  if original_test_labels[i] != predicted_test_labels[i]:
    if predicted_test_probs[i] > tau or predicted_test_probs[i] < (1 - tau):
      final_test_labels.append(predicted_test_labels[i])
    else:
      final_test_labels.append(original_test_labels[i])
  else:
    final_test_labels.append(original_test_labels[i])

final_test_labels = np.asarray(final_test_labels)

np.savetxt('train_labels.csv', final_train_labels, delimiter=',')
np.savetxt('test_labels.csv', final_test_labels, delimiter=',')