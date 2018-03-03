# Text Helper Functions
# ---------------------------------------
#
# We pull out text helper functions to reduce redundant code

import collections
import os
import string
import tarfile

import numpy as np
import requests


# Normalize text
def normalize_text(texts, stops):
  texts = [x.lower() for x in texts]
  texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
  texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
  texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]
  texts = [' '.join(x.split()) for x in texts]
  return texts


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
    for word in sentence.split():
      if word in word_dict:
        word_ix = word_dict[word]
      else:
        word_ix = 0
      sentence_data.append(word_ix)
    data.append(sentence_data)
  return data


# Generate data randomly (N words behind, target, N words ahead)
def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
  # Fill up data batch
  batch_data = []
  label_data = []
  while len(batch_data) < batch_size:
    # select random sentence to start
    rand_sentence_ix = int(np.random.choice(len(sentences), size=1))
    rand_sentence = sentences[rand_sentence_ix]
    # Generate consecutive windows to look at
    window_sequences = [rand_sentence[max((ix - window_size), 0):(ix + window_size + 1)] for ix, _ in
                        enumerate(rand_sentence)]
    # Denote which element of each window is the center word of interest
    label_indices = [ix if ix < window_size else window_size for ix, _ in enumerate(window_sequences)]

    # Pull out center word of interest for each window and create a tuple for each window
    if method == 'skip_gram':
      batch_and_labels = [(words[i], words[:i] + words[(i + 1):]) for words, i in zip(window_sequences, label_indices)]
      # Make it in to a big list of tuples (target word, surrounding word)
      tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
      batch, labels = [list(x) for x in zip(*tuple_data)]
    elif method == 'cbow':
      batch_and_labels = [(words[:i] + words[(i + 1):], words[i]) for words, i in zip(window_sequences, label_indices)]
      # Only keep windows with consistent 2*window_size
      batch_and_labels = [(x, y) for x, y in batch_and_labels if len(x) == 2 * window_size]
      batch, labels = [list(x) for x in zip(*batch_and_labels)]
    elif method == 'doc2vec':
      # For doc2vec we keep LHS window only to predict target word
      batch_and_labels = [(rand_sentence[i:i + window_size], rand_sentence[i + window_size]) for i in
                          range(0, len(rand_sentence) - window_size)]
      batch, labels = [list(x) for x in zip(*batch_and_labels)]
      # Add document index to batch!! Remember that we must extract the last index in batch for the doc-index
      batch = [x + [rand_sentence_ix] for x in batch]
    else:
      raise ValueError('Method {} not implemented yet.'.format(method))

    # (Batch, labels) tuples:
    #
    # Skip-Gram:
    #
    #       sentence = [5, 21, 33, 4406, 25, ...]
    #       pairs = <center, context>
    #       window size = 2
    #
    #       center=5   (5, 21), (5, 33),
    #       center=21  (21, 5), (21, 33), (21, 4406),
    #       center=33  (33, 5), (33, 21), (33, 4406), (33, 25), ...
    #
    #
    # CBOW:
    #
    #       sentence = [1388, 0, 0, 1783, 203, 175, 85, 0, 69, 1598]
    #       pairs = <full-context, center>
    #       window size = 3
    #
    #       center=1783  ([1388, 0, 0, 203, 175, 85], 1783)
    #       center=203   ([0, 0, 1783, 175, 85, 0], 203)
    #       center=175   ([0, 1783, 203, 85, 0, 69], 175)
    #       center=85    ([1783, 203, 175, 0, 69, 1598], 85)
    #
    #
    # Doc2Vec (DMPV or PV-DM = paragraph vector - distributed memory, similar to CBOW):
    #
    #       sentence = [17, 1026, 1475, 18, 13, 2, 2608, 8, 72, 762, 51]
    #       pairs = <left-context + sentence_id, center>
    #       sentence_id = 3723
    #       window size = 3
    #
    #       center=18    ([17, 1026, 1475, 3723], 18)
    #       center=13    ([1026, 1475, 18, 3723], 13)
    #       center=2     ([1475, 18, 13, 3723], 2)
    #       center=2608  ([18, 13, 2, 3723], 2608)
    #       center=8     ([13, 2, 2608, 3723], 8)
    #       center=72    ([2, 2608, 8, 3723], 72)
    #       center=762   ([2608, 8, 72, 3723], 762)
    #       center=51    ([8, 72, 762, 3723], 51)
    #
    # rem:  though this implementation uses only left-context, it's not mandated anyhow by PV-DM.
    #       In "Machine Learning and Knowledge Extraction (2017)", authors show that PV-DM can use full
    #       context (left and right) just like CBOW does. E.g., Gensim implementation uses full window
    #       (see `gensim.models.doc2vec.train_document_dm`).
    #
    # Doc2Vec (DBOW or PV-DBOW = paragraph vector - distributed bag of words, similar to Skip-Gram).
    #
    #       Not implemented here, but the idea is:
    #       There's one vector for the text, which alone is used to predict each individual word.
    #       So while word-vectors still get randomly initialized, they're not updated during training at all,
    #       and are still at their random initial values at the end of training.
    #
    #       Now, the DBOW training is very analogous to the Word2Vec "skip-gram" mode,
    #       but using vector(s) for the text-as-a-whole to predict target words,
    #       rather than just vector(s) for nearby-words...
    #       so it is very easy to combine with skip-gram word training, if you need the word-vectors too.
    #
    # rem:  PV-DBOW does not use a sliding context window, 
    #       instead all words from the sentence are considered as context.
    #

    # extract batch and labels
    batch_data.extend(batch[:batch_size])
    label_data.extend(labels[:batch_size])

  # Trim batch and label at the end
  batch_data = batch_data[:batch_size]
  label_data = label_data[:batch_size]

  # Convert to numpy array
  batch_data = np.array(batch_data)
  label_data = np.expand_dims(label_data, axis=1)

  return batch_data, label_data


# Load the movie review data
# Check if data was downloaded, otherwise download it and save for future use
def load_movie_data(dir):
  pos_file = os.path.join(dir, 'rt-polaritydata', 'rt-polarity.pos')
  neg_file = os.path.join(dir, 'rt-polaritydata', 'rt-polarity.neg')

  # Check if files are already downloaded
  if not os.path.exists(os.path.join(dir, 'rt-polaritydata')):
    movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'

    # Save tar.gz file
    req = requests.get(movie_data_url, stream=True)
    with open('temp_movie_review_temp.tar.gz', 'wb') as f:
      for chunk in req.iter_content(chunk_size=1024):
        if chunk:
          f.write(chunk)
          f.flush()
    # Extract tar.gz file into temp folder
    tar = tarfile.open('temp_movie_review_temp.tar.gz', "r:gz")
    tar.extractall(path=dir)
    tar.close()

  pos_data = []
  with open(pos_file, 'r', encoding='latin-1') as f:
    for line in f:
      pos_data.append(line.encode('ascii', errors='ignore').decode())
  pos_data = [x.rstrip() for x in pos_data]

  neg_data = []
  with open(neg_file, 'r', encoding='latin-1') as f:
    for line in f:
      neg_data.append(line.encode('ascii', errors='ignore').decode())
  neg_data = [x.rstrip() for x in neg_data]

  texts = pos_data + neg_data
  target = [1] * len(pos_data) + [0] * len(neg_data)

  return texts, target
