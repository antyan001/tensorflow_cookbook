# Doc2Vec Model
# ---------------------------------------
#
# In this example, we will download and preprocess the movie
#  review data.
#
# From this data set we will compute/fit a Doc2Vec model to get
# Document vectors.  From these document vectors, we will split the
# documents into train/test and use these doc vectors to do sentiment
# analysis on the movie review dataset.

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import text_helpers

os.chdir(os.path.dirname(os.path.realpath(__file__)))


########################################################################################################################
# Data
########################################################################################################################


# Make a saving directory if it doesn't exist
tmp_dir = 'temp'
if not os.path.exists(tmp_dir):
  os.makedirs(tmp_dir)

# Declare model parameters
batch_size = 500
vocabulary_size = 7500
generations = 100000
learning_rate = 0.001

embedding_size = 200  # Word embedding size
doc_embedding_size = 100  # Document embedding size
concatenated_size = embedding_size + doc_embedding_size

num_sampled = int(batch_size / 2)  # Number of negative examples to sample.
window_size = 3  # How many words to consider to the left.

# Add checkpoints to training
save_embeddings_every = 5000
print_valid_every = 5000
print_loss_every = 100

# We pick a few test words for validation.
valid_words = ['love', 'hate', 'happy', 'sad', 'man', 'woman']

texts, target = text_helpers.load_movie_data(tmp_dir)
texts = text_helpers.normalize_text(texts, stops=[])
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > window_size]
texts = [x for x in texts if len(x.split()) > window_size]
assert (len(target) == len(texts))

# Build our data set and dictionaries
word_dictionary = text_helpers.build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_helpers.text_to_numbers(texts, word_dictionary)

# Get validation word keys
valid_examples = [word_dictionary[x] for x in valid_words]


########################################################################################################################
# Model: Doc2Vec PV-DM (paragraph vector - distributed memory)
########################################################################################################################


# From https://arxiv.org/pdf/1607.05368.pdf
#
# There are two approaches within word2vec: skip-gram (“sg”) and cbow. In skip-gram, the input is a word
# (i.e. vwI is a vector of one word) and the output is a context word. For each input word,
# the number of left or right context words to predict is defined by the window size hyperparameter.
# cbow is different to skip-gram in one aspect: the input consists of multiple words that are combined
# via vector addition to predict the context word (i.e. vwI is a summed vector of several words).
#
# doc2vec is an extension to word2vec for learning document embeddings (Le and Mikolov, 2014).
# There are two approaches within doc2vec: dbow and dmpv.
#   dbow works in the same way as skip-gram, except that the input is replaced by a special token representing
# the document (i.e. vwI is a vector representing the document). In this architecture, the order of words in the
# document is ignored; hence the name distributed bag of words.
#   dmpv works in a similar way to cbow. For the input, dmpv introduces an additional document token in addition
# to multiple target words. Unlike cbow, however, these vectors are not summed but concatenated
# (i.e. vwI is a concatenated vector containing the document token and several target words).
# The objective is again to predict a context word given the concatenated document and word vectors..

doc2vec_x = tf.placeholder(tf.int32, shape=[None, window_size + 1])  # plus 1 for doc index
doc2vec_y = tf.placeholder(tf.int32, shape=[None, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Define Embeddings:
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
doc_embeddings = tf.Variable(tf.random_uniform([len(texts), doc_embedding_size], -1.0, 1.0))

# Lookup and add together the word embedding in window
embed = tf.zeros([batch_size, embedding_size])
for element in range(window_size):
  embed += tf.nn.embedding_lookup(embeddings, doc2vec_x[:, element])

doc_indices = tf.slice(doc2vec_x, [0, window_size], [batch_size, 1])
doc_embed = tf.nn.embedding_lookup(doc_embeddings, doc_indices)

# Concatenate embeddings (here can also be the sum)
final_embed = tf.concat(axis=1, values=[embed, tf.squeeze(doc_embed)])

# Get loss from prediction
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, concatenated_size],
                                              stddev=1.0 / np.sqrt(concatenated_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=doc2vec_y,
                                     inputs=final_embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Cosine similarity between words
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

saver = tf.train.Saver({'embeddings': embeddings, 'doc_embeddings': doc_embeddings})
embeddings_checkpoint_path = os.path.join(os.getcwd(), tmp_dir, 'doc2vec_movie_embeddings.ckpt')
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  print('Starting Training Doc2Vec')
  loss_vec = []
  loss_x_vec = []
  for i in range(generations):
    batch_inputs, batch_labels = text_helpers.generate_batch_data(text_data, batch_size,
                                                                  window_size, method='doc2vec')
    feed_dict = {doc2vec_x: batch_inputs, doc2vec_y: batch_labels}

    # Run the train step
    sess.run(train_step, feed_dict=feed_dict)

    # Return the loss
    if (i + 1) % print_loss_every == 0:
      loss_val = sess.run(loss, feed_dict=feed_dict)
      loss_vec.append(loss_val)
      loss_x_vec.append(i + 1)
      print('Loss at step {} : {}'.format(i + 1, loss_val))

    # Validation: Print some random words and top 5 related words
    if (i + 1) % print_valid_every == 0:
      sim = sess.run(similarity, feed_dict=feed_dict)
      for j in range(len(valid_words)):
        valid_word = word_dictionary_rev[valid_examples[j]]
        top_k = 5  # number of nearest neighbors
        nearest = (-sim[j, :]).argsort()[1:top_k + 1]
        log_str = "Nearest to {}:".format(valid_word)
        for k in range(top_k):
          close_word = word_dictionary_rev[nearest[k]]
          log_str = '{} {},'.format(log_str, close_word)
        print(log_str)

    # Save dictionary + embeddings
    if (i + 1) % save_embeddings_every == 0:
      # Save vocabulary dictionary
      with open(os.path.join(tmp_dir, 'movie_vocab.pkl'), 'wb') as f:
        pickle.dump(word_dictionary, f)

      # Save embeddings
      save_path = saver.save(sess, embeddings_checkpoint_path)
      print('Model saved in file: {}'.format(save_path))


########################################################################################################################
# Data preparation: Sentiment analysis using Doc2Vec embeddings
########################################################################################################################


max_words = 20
logistic_batch_size = 500

# Split dataset into train and test sets
# Need to keep the indices sorted to keep track of document index
train_indices = np.sort(np.random.choice(len(target), round(0.8 * len(target)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(target))) - set(train_indices))))
texts_train = [x for ix, x in enumerate(texts) if ix in train_indices]
texts_test = [x for ix, x in enumerate(texts) if ix in test_indices]
target_train = np.array([x for ix, x in enumerate(target) if ix in train_indices])
target_test = np.array([x for ix, x in enumerate(target) if ix in test_indices])

# Convert texts to lists of indices
text_data_train = np.array(text_helpers.text_to_numbers(texts_train, word_dictionary))
text_data_test = np.array(text_helpers.text_to_numbers(texts_test, word_dictionary))

# Pad/crop movie reviews to specific length
text_data_train = np.array([x[0:max_words] for x in [y + [0] * max_words for y in text_data_train]])
text_data_test = np.array([x[0:max_words] for x in [y + [0] * max_words for y in text_data_test]])


########################################################################################################################
# Model
########################################################################################################################


sentim_x = tf.placeholder(tf.int32, shape=[None, max_words + 1])  # plus 1 for doc index
sentim_y = tf.placeholder(tf.int32, shape=[None, 1])

# Define logistic embedding lookup (needed if we have two different batch sizes)
# Add together element embeddings in window:
sentim_embed = tf.zeros([logistic_batch_size, embedding_size])
for element in range(max_words):
  sentim_embed += tf.nn.embedding_lookup(embeddings, sentim_x[:, element])

sentim_doc_indices = tf.slice(sentim_x, [0, max_words], [logistic_batch_size, 1])
sentim_doc_embed = tf.nn.embedding_lookup(doc_embeddings, sentim_doc_indices)
sentim_final_embed = tf.concat(axis=1, values=[sentim_embed, tf.squeeze(sentim_doc_embed)])

# Define model:
W = tf.Variable(tf.random_normal(shape=[concatenated_size, 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))
model_output = tf.add(tf.matmul(sentim_final_embed, W), b)

# Declare loss function (Cross Entropy loss)
logistic_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output,
                                                                       labels=tf.cast(sentim_y, tf.float32)))
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, tf.cast(sentim_y, tf.float32)), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)
sentim_train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(logistic_loss, var_list=[W, b])

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver.restore(sess, embeddings_checkpoint_path)

  print('Starting Logistic Model Training')
  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []
  i_data = []
  for i in range(10000):
    rand_index = np.random.choice(text_data_train.shape[0], size=logistic_batch_size)
    rand_x = text_data_train[rand_index]
    # Append review index at the end of text data
    rand_x_doc_indices = train_indices[rand_index]
    rand_x = np.hstack((rand_x, np.transpose([rand_x_doc_indices])))
    rand_y = np.transpose([target_train[rand_index]])

    feed_dict = {sentim_x: rand_x, sentim_y: rand_y}
    sess.run(sentim_train_op, feed_dict=feed_dict)

    # Only record loss and accuracy every 100 generations
    if (i + 1) % 100 == 0:
      rand_index_test = np.random.choice(text_data_test.shape[0], size=logistic_batch_size)
      rand_x_test = text_data_test[rand_index_test]
      # Append review index at the end of text data
      rand_x_doc_indices_test = test_indices[rand_index_test]
      rand_x_test = np.hstack((rand_x_test, np.transpose([rand_x_doc_indices_test])))
      rand_y_test = np.transpose([target_test[rand_index_test]])

      test_feed_dict = {sentim_x: rand_x_test, sentim_y: rand_y_test}

      i_data.append(i + 1)

      train_loss_temp = sess.run(logistic_loss, feed_dict=feed_dict)
      train_loss.append(train_loss_temp)

      test_loss_temp = sess.run(logistic_loss, feed_dict=test_feed_dict)
      test_loss.append(test_loss_temp)

      train_acc_temp = sess.run(accuracy, feed_dict=feed_dict)
      train_acc.append(train_acc_temp)

      test_acc_temp = sess.run(accuracy, feed_dict=test_feed_dict)
      test_acc.append(test_acc_temp)
    if (i + 1) % 500 == 0:
      acc_and_loss = [i + 1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
      acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
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
