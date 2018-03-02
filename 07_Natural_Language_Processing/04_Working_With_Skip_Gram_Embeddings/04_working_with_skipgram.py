# Word2Vec: Skipgram Model
# ---------------------------------------
#
# In this example, we will download and preprocess the movie
#  review data.
#
# From this data set we will compute/fit the skipgram model of
#  the Word2Vec Algorithm
#
# Skipgram: based on predicting the surrounding words from the
#  Ex sentence "the cat in the hat"
#  context word:  ["hat"]
#  target words: ["the", "cat", "in", "the"]
#  context-target pairs:
#    ("hat", "the"), ("hat", "cat"), ("hat", "in"), ("hat", "the")


import collections
import os
import string
import tarfile

import numpy as np
import requests
import tensorflow as tf
from nltk.corpus import stopwords

import nltk
nltk.data.path.append('/home/maxim/bin/nltk')

os.chdir(os.path.dirname(os.path.realpath(__file__)))


########################################################################################################################
# Data
########################################################################################################################


# Declare model parameters
batch_size = 100
embedding_size = 200
vocabulary_size = 10000
generations = 10000
print_loss_every = 2000

num_sampled = int(batch_size / 2)  # Number of negative examples to sample.
window_size = 2  # How many words to consider left and right.

# Declare stop words
stops = stopwords.words('english')

# We pick five test words. We are expecting synonyms to appear
print_valid_every = 5000
valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']
# Later we will have to transform these into indices

# Load the movie review data
# Check if data was downloaded, otherwise download it and save for future use
def load_movie_data():
  save_folder_name = 'temp'
  pos_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.pos')
  neg_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.neg')

  # Check if files are already downloaded
  if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
    movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

    # Save tar.gz file
    req = requests.get(movie_data_url, stream=True)
    with open('temp_movie_review_temp.tar.gz', 'wb') as f:
      for chunk in req.iter_content(chunk_size=1024):
        if chunk:
          f.write(chunk)
          f.flush()
    # Extract tar.gz file into temp folder
    tar = tarfile.open('temp_movie_review_temp.tar.gz', 'r:gz')
    tar.extractall(path='temp')
    tar.close()

  with open(pos_file, 'r', errors='replace') as file_:
    pos_data = file_.readlines()
    pos_data = [x.rstrip() for x in pos_data]

  with open(neg_file, 'r', errors='replace') as file_:
    neg_data = file_.readlines()
    neg_data = [x.rstrip() for x in neg_data]

  texts = pos_data + neg_data
  target = [1] * len(pos_data) + [0] * len(neg_data)

  return texts, target

# Load the movie review data:
# - texts is a list of strings (sentences)
# - target is a list of 0 or 1 (unused in this example).
texts, _ = load_movie_data()

# Normalize text
def normalize_text(texts, stops):
  texts = [x.lower() for x in texts]
  texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
  texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
  texts = [' '.join([word for word in x.split() if word not in stops]) for x in texts]
  texts = [' '.join(x.split()) for x in texts]
  return texts

texts = normalize_text(texts, stops)
texts = [x for x in texts if len(x.split()) > 2]

# Build dictionary of words
def build_dictionary(sentences, vocabulary_size):
  # Turn sentences (list of strings) into lists of words
  split_sentences = [s.split() for s in sentences]
  words = [x for sublist in split_sentences for x in sublist]

  # Initialize list of [word, word_count] for each word, starting with unknown
  count = [['RARE', -1]]

  # Now add most frequent words, limited to the N-most frequent (N=vocabulary size)
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

  # Now create the dictionary
  word_dict = {}
  # For each word, that we want in the dictionary, add it, then make it
  # the value of the prior dictionary length
  for word, word_count in count:
    word_dict[word] = len(word_dict)

  return word_dict

# Turn text data into lists of integers from dictionary
def text_to_numbers(sentences, word_dict):
  # Initialize the returned data
  data = []
  for sentence in sentences:
    sentence_data = []
    # For each word, either use selected index or rare word index
    for word in sentence.split(' '):
      if word in word_dict:
        word_ix = word_dict[word]
      else:
        word_ix = 0
      sentence_data.append(word_ix)
    data.append(sentence_data)
  return data

# Build our data set and dictionaries
word_dictionary = build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_numbers(texts, word_dictionary)

# Get validation word keys
valid_examples = [word_dictionary[x] for x in valid_words]

# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
  # Fill up data batch
  batch_data = []
  label_data = []
  while len(batch_data) < batch_size:
    # select random sentence to start
    rand_sentence = np.random.choice(sentences)
    # Generate consecutive windows to look at
    window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)] for ix, x in
                        enumerate(rand_sentence)]
    # Denote which element of each window is the center word of interest
    label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequences)]

    # Pull out center word of interest for each window and create a tuple for each window
    if method == 'skip_gram':
      batch_and_labels = [(words[i], words[:i] + words[(i + 1):]) for words, i in zip(window_sequences, label_indices)]
      # Make it in to a big list of tuples (target word, surrounding word)
      tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
    elif method == 'cbow':
      batch_and_labels = [(words[:i] + words[(i + 1):], words[i]) for words, i in zip(window_sequences, label_indices)]
      # Make it in to a big list of tuples (target word, surrounding word)
      tuple_data = [(x_, y) for x, y in batch_and_labels for x_ in x]
    else:
      raise ValueError('Method {} not implemented yet.'.format(method))

    # `tuple_data` duplicates either x or y to make the list of (x, y) samples.
    # By the way, the example from the book is wrong (the author tries to describe SG, but talks about CBOW).
    #
    # SG:   (5, 21), (5, 33),
    #       (21, 5), (21, 33), (21, 4406),
    #       (33, 5), (33, 21), (33, 4406), (33, 25), ...
    #       <center, context>
    #
    # CBOW: (121, 276), (1, 276),
    #       (276, 121), (1, 121), (7146, 121),
    #       (276, 1), (121, 1), (7146, 1), (716, 1), ...
    #       <context, center>

    # extract batch and labels
    batch, labels = [list(x) for x in zip(*tuple_data)]
    batch_data.extend(batch[:batch_size])
    label_data.extend(labels[:batch_size])

  # Trim batch and label at the end
  batch_data = batch_data[:batch_size]
  label_data = label_data[:batch_size]

  # Convert to numpy array
  batch_data = np.array(batch_data)
  label_data = np.expand_dims(label_data, axis=1)

  return batch_data, label_data


########################################################################################################################
# Model
# The input pairs may differ (either x or y is reused multiple times).
# The model is pure Skip-Gram. In CBOW we would need to add up all embeddings.
#
# The input indices are passed through an input embedding layer,
# then the embeddings go through NCE (with an output embedding layer) to compute the loss.
# It is a classification task to predict one word index given the other.
########################################################################################################################


# Create data/target placeholders
x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Lookup the word embedding
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, x_inputs)

# NCE loss parameters
# NOTE: `nce_weights` has the same shape as `embeddings`: the embedding matrix on the other side.
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                              stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Get loss from prediction
loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                     biases=nce_biases,
                                     labels=y_target,
                                     inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

# Cosine similarity between words
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

with tf.Session() as sess:
  # Add variable initializer.
  sess.run(tf.global_variables_initializer())

  # Run the skip gram model.
  loss_vec = []
  loss_x_vec = []
  for i in range(generations):
    batch_inputs, batch_labels = generate_batch_data(text_data, batch_size, window_size)
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}

    # Run the train step
    sess.run(optimizer, feed_dict=feed_dict)

    # Return the loss
    if (i + 1) % print_loss_every == 0:
      loss_val = sess.run(loss, feed_dict=feed_dict)
      loss_vec.append(loss_val)
      loss_x_vec.append(i + 1)
      print("Loss at step {} : {}".format(i + 1, loss_val))

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
          log_str = "%s %s," % (log_str, close_word)
        print(log_str)
