# Implementing TF-IDF
# ---------------------------------------
#
# Here we implement TF-IDF,
#  (Text Frequency - Inverse Document Frequency)
#  for the spam-ham text data.
#
# We will use a hybrid approach of encoding the texts
#  with sci-kit learn's TFIDF vectorizer.  Then we will
#  use the regular TensorFlow logistic algorithm outline.

import csv
import io
import os
import string
from zipfile import ZipFile

import matplotlib.pyplot as plt
import nltk
import numpy as np
import requests
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer


# Data

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

# Define tokenizer
def tokenizer(text):
  nltk.data.path.append('/home/maxim/bin/nltk')
  words = nltk.word_tokenize(text)
  return words

batch_size = 200
max_features = 1000

# Create TF-IDF of texts
tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words='english', max_features=max_features)
sparse_tfidf_texts = tfidf.fit_transform(texts)

# Split up data set into train/test
train_indices = np.random.choice(sparse_tfidf_texts.shape[0], int(0.8 * sparse_tfidf_texts.shape[0]), replace=False)
test_indices = np.array(list(set(range(sparse_tfidf_texts.shape[0])) - set(train_indices)))
texts_train = sparse_tfidf_texts[train_indices]
texts_test = sparse_tfidf_texts[test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# Model

x = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Logistic model (sigmoid in loss function)
W = tf.Variable(tf.random_normal(shape=[max_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
model_output = tf.add(tf.matmul(x, W), b)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y))
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)
train_step = tf.train.GradientDescentOptimizer(0.0025).minimize(loss)

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  # Start Logistic Regression
  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []
  i_data = []
  for i in range(10000):
    rand_index = np.random.choice(texts_train.shape[0], size=batch_size)
    rand_x = texts_train[rand_index].todense()
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict={x: rand_x, y: rand_y})

    if (i + 1) % 100 == 0:
      i_data.append(i + 1)
      train_loss_temp = sess.run(loss, feed_dict={x: rand_x, y: rand_y})
      train_loss.append(train_loss_temp)

      test_loss_temp = sess.run(loss, feed_dict={x: texts_test.todense(), y: np.transpose([target_test])})
      test_loss.append(test_loss_temp)

      train_acc_temp = sess.run(accuracy, feed_dict={x: rand_x, y: rand_y})
      train_acc.append(train_acc_temp)

      test_acc_temp = sess.run(accuracy, feed_dict={x: texts_test.todense(), y: np.transpose([target_test])})
      test_acc.append(test_acc_temp)

    if (i + 1) % 500 == 0:
      acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
      acc_and_loss = [np.round(v, 2) for v in acc_and_loss]
      print('Generation # {}. Train Loss (Test Loss): {:.2f} ({:.2f}). Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

# Plot loss over time
plt.plot(i_data, train_loss, 'k-', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

# Plot train and test accuracy
plt.plot(i_data, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Set Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()
