#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Implementing an RNN in TensorFlow
# ----------------------------------
#
# We implement an RNN in TensorFlow to predict spam/ham from texts
#

from __future__ import print_function
import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile

# Download or open data
data_dir = 'temp'
data_file = 'text_data.txt'
if not os.path.exists(data_dir):
  os.makedirs(data_dir)

if not os.path.isfile(os.path.join(data_dir, data_file)):
  zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
  r = requests.get(zip_url)
  z = ZipFile(io.BytesIO(r.content))
  file = z.read('SMSSpamCollection')
  # Format Data
  text_data = file.decode(errors='ignore')
  text_data = text_data.encode('ascii', errors='ignore')
  text_data = text_data.decode().split('\n')

  # Save data to text file
  with open(os.path.join(data_dir, data_file), 'w') as file_conn:
    for text in text_data:
      file_conn.write("{}\n".format(text))
else:
  # Open data from text file
  text_data = []
  with open(os.path.join(data_dir, data_file), 'r') as file_conn:
    for row in file_conn:
      text_data.append(row)
  text_data = text_data[:-1]

# Create a text cleaning function
def clean_text(text_string):
  text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
  text_string = " ".join(text_string.split())
  text_string = text_string.lower()
  return text_string

text_data = [x.split('\t') for x in text_data if len(x) >= 1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]
text_data_train = [clean_text(x) for x in text_data_train]
print('Data examples:')
for text, target in zip(text_data_train[:15], text_data_target[:15]):
  print('%4s: %s' % (target, text))
print()


# Hyper-parameters
epochs = 80
batch_size = 250
max_sequence_length = 25
rnn_hidden_size = 20
embedding_size = 50
min_word_frequency = 10
learning_rate = 0.0005

# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                     min_frequency=min_word_frequency)
text_processed = list(vocab_processor.fit_transform(text_data_train))
print('Convert text to idx:')
for i in range(3):
  print(text_data_train[i], '->', np.array2string(text_processed[i], max_line_width=100, separator=' '))
vocab_mapping = vocab_processor.vocabulary_._mapping
for i in range(5):
  item = text_data_train[0].split(' ')[i]
  print(item, '->', vocab_mapping.get(item, '?'))
print()


# Shuffle and split data
text_processed = np.array(text_processed)
text_data_target = np.array([1 if x == 'ham' else 0 for x in text_data_target])
shuffled_ix = np.random.permutation(np.arange(len(text_data_target)))
x_shuffled = text_processed[shuffled_ix]
y_shuffled = text_data_target[shuffled_ix]

# Split train/test set
ix_cutoff = int(len(y_shuffled) * 0.80)
x_train, x_test = x_shuffled[:ix_cutoff], x_shuffled[ix_cutoff:]
y_train, y_test = y_shuffled[:ix_cutoff], y_shuffled[ix_cutoff:]
vocab_size = len(vocab_processor.vocabulary_)
print("Vocabulary Size: {:d}".format(vocab_size))
print("80-20 train/ test split: train={:d}, test={:d}".format(len(y_train), len(y_test)))
print()


# Model

x_data = tf.placeholder(tf.int32, [None, max_sequence_length])
y_output = tf.placeholder(tf.int32, [None])
dropout_keep_prob = tf.placeholder_with_default(1.0, shape=())

# Create embedding
embedding_mat = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
embedding_output = tf.nn.embedding_lookup(embedding_mat, x_data)
# embedding_output_expanded = tf.expand_dims(embedding_output, -1)

# Define the RNN cell
if tf.__version__[0] >= '1':
  cell = tf.contrib.rnn.BasicRNNCell(num_units=rnn_hidden_size)
else:
  cell = tf.nn.rnn_cell.BasicRNNCell(num_units=rnn_hidden_size)

output, state = tf.nn.dynamic_rnn(cell, embedding_output, dtype=tf.float32)
output = tf.nn.dropout(output, dropout_keep_prob)

# Get output of RNN sequence
# output = (?, max_sequence_length, rnn_hidden_size)
# output_last = (?, rnn_hidden_size)
output = tf.transpose(output, [1, 0, 2])
output_last = tf.gather(output, int(output.get_shape()[0]) - 1)

# Final logits for binary classification:
# logits = (?, 2)
weight = tf.Variable(tf.truncated_normal([rnn_hidden_size, 2], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[2]))
logits = tf.matmul(output_last, weight) + bias

# Loss function
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_output)
loss = tf.reduce_mean(cross_entropy)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(y_output, tf.int64)), tf.float32))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_step = optimizer.minimize(loss)

# Start training
train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  for epoch in range(epochs):
    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train) / batch_size) + 1

    for i in range(num_batches):
      min_ix = i * batch_size
      max_ix = np.min([len(x_train), ((i + 1) * batch_size)])
      x_train_batch = x_train[min_ix:max_ix]
      y_train_batch = y_train[min_ix:max_ix]
      sess.run(train_step, feed_dict={x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob: 0.5})

    # Run loss and accuracy for training
    train_loss_val, train_acc_val = sess.run([loss, accuracy], feed_dict={x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob: 0.5})
    train_loss.append(train_loss_val)
    train_accuracy.append(train_acc_val)

    # Run Eval Step
    test_loss_val, test_acc_val = sess.run([loss, accuracy], feed_dict={x_data: x_test, y_output: y_test})
    test_loss.append(test_loss_val)
    test_accuracy.append(test_acc_val)
    print('Epoch: {}, Train loss: {:.3}, Train Acc: {:.3}, Test Loss: {:.3}, Test Acc: {:.3}'
          .format(epoch + 1, train_loss_val, train_acc_val, test_loss_val, test_acc_val))

# Plot loss over time
epoch_seq = np.arange(1, epochs + 1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
