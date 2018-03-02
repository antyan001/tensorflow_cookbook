# Working with Bag of Words
# ---------------------------------------
#
# In this example, we will download and preprocess the ham/spam
#  text data.  We will then use a one-hot-encoding to make a
#  bag of words set of features to use in logistic regression.
#
# We will use these one-hot-vectors for logistic regression to
#  predict if a text is spam or ham.

import csv
import io
import os
import string
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from tensorflow.contrib import learn


########################################################################################################################
# Data
########################################################################################################################


# Check if data was downloaded, otherwise download it and save for future use
save_file_name = os.path.join('temp', 'temp_spam_data.csv')

# Create directory if it doesn't exist
if not os.path.exists('temp'):
  os.makedirs('temp')

if os.path.isfile(save_file_name):
  text_data = []
  with open(save_file_name, 'r') as temp_output_file:
    reader = csv.reader(temp_output_file)
    for row in reader:
      text_data.append(row)
else:
  zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
  r = requests.get(zip_url)
  with ZipFile(io.BytesIO(r.content)) as z:
    with z.open('SMSSpamCollection', 'r') as file_:
      text_data = file_.readlines()
      text_data = [line.decode('utf-8') for line in text_data]
      text_data = [x.split('\t') for x in text_data if len(x) >= 1]
  with open(save_file_name, 'w') as temp_output_file:
    writer = csv.writer(temp_output_file)
    writer.writerows(text_data)

texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]

# Relabel 'spam' as 1, 'ham' as 0
target = [1 if x == 'spam' else 0 for x in target]

# Normalize text
texts = [x.lower() for x in texts]
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
texts = [' '.join(x.split()) for x in texts]

# Plot histogram of text lengths
text_lengths = [len(x.split()) for x in texts]
text_lengths = [x for x in text_lengths if x < 50]
plt.hist(text_lengths, bins=25)
plt.title('Histogram of # of Words in Texts')
plt.show()

# Choose max text word length at 25
sentence_size = 25
min_word_freq = 3

# Setup vocabulary processor
vocab_processor = learn.preprocessing.VocabularyProcessor(sentence_size, min_frequency=min_word_freq)

# Have to fit transform to get length of unique words.
transform = vocab_processor.transform(texts)
embedding_size = len([x for x in transform])
print('embedding_size=%d' % embedding_size)

# Split up data set into train/test
train_indices = np.random.choice(len(texts), int(len(texts) * 0.8), replace=False)
test_indices = np.array(list(set(range(len(texts))) - set(train_indices)))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = [x for ix, x in enumerate(target) if ix in train_indices]
target_test = [x for ix, x in enumerate(target) if ix in test_indices]


########################################################################################################################
# Model
# Bag-of-words: feature vector consists of the number of times each token is present:
# [10, 0, 0, 0, 0, 2, 0, 0, 0, 1, ... ] -> the words are 0, 5 and 9, in any order.
########################################################################################################################


# hard-coded batch_size=1
x = tf.placeholder(shape=[sentence_size], dtype=tf.int32)
y = tf.placeholder(shape=[1, 1], dtype=tf.float32)

# Text-vocab embedding (id matrix for one-hot-encoding)
identity_matrix = tf.diag(tf.ones(shape=[embedding_size]))
x_embed = tf.nn.embedding_lookup(identity_matrix, x)
x_embed = tf.Print(x_embed, data=[x_embed], message='Embedding', summarize=10, first_n=0)
x_col_sums = tf.reduce_sum(x_embed, axis=0)
x_col_sums = tf.Print(x_col_sums, data=[x_col_sums], message='x-sum', summarize=10, first_n=0)

# Logistic regression model (one hidden layer)
W = tf.Variable(tf.random_normal(shape=[embedding_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
x_col_sums_2D = tf.expand_dims(x_col_sums, 0)
model_output = tf.add(tf.matmul(x_col_sums_2D, W), b)

# Declare loss function (Cross Entropy loss)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y))
prediction = tf.sigmoid(model_output)
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  # Start Logistic Regression
  print('Starting training over %d sentences.' % len(texts_train))
  loss_vec = []
  train_acc_all = []
  train_acc_avg = []
  for ix, t in enumerate(vocab_processor.fit_transform(texts_train)):
    y_data = [[target_train[ix]]]

    _, temp_loss = sess.run([train_step, loss], feed_dict={x: t, y: y_data})
    loss_vec.append(temp_loss)

    if (ix + 1) % 10 == 0:
      print('Training observation #%d: Loss = %.5f' % (ix + 1, temp_loss))

    # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    temp_pred = sess.run(prediction, feed_dict={x: t, y: y_data})

    train_acc_temp = target_train[ix] == np.round(temp_pred)
    train_acc_all.append(train_acc_temp)
    if len(train_acc_all) >= 50:
      train_acc_avg.append(np.mean(train_acc_all[-50:]))

  # Get test set accuracy
  print('Getting test set accuracy for %d sentences.' % len(texts_test))
  test_acc_all = []
  for ix, t in enumerate(vocab_processor.fit_transform(texts_test)):
    y_data = [[target_test[ix]]]

    if (ix + 1) % 50 == 0:
      print('Test observation #' + str(ix + 1))

      # Keep trailing average of past 50 observations accuracy
    # Get prediction of single observation
    [[temp_pred]] = sess.run(prediction, feed_dict={x: t, y: y_data})
    # Get True/False if prediction is accurate
    test_acc_temp = target_test[ix] == np.round(temp_pred)
    test_acc_all.append(test_acc_temp)

print('\nOverall Test Accuracy: {}'.format(np.mean(test_acc_all)))

# Plot training accuracy over time
plt.plot(range(len(train_acc_avg)), train_acc_avg, 'k-', label='Train Accuracy')
plt.title('Avg Training Acc Over Past 50 Generations')
plt.xlabel('Generation')
plt.ylabel('Training Accuracy')
plt.show()
