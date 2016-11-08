''' 
This example demonstrate the use of item2vec[1] on the 1M MovieLens dataset[2].

References:
[1] Oren Barkan and Noam Koenigstein,
    "Item2Vec: Neural Item Embedding for Collaborative Filtering",
    https://arxiv.org/abs/1603.04259
  
[2] http://grouplens.org/datasets/movielens/1m

Results for Aladdin (after 25 epochs):

Seed:  588 Aladdin (1992) Animation|Children's|Comedy|Musical

  ID  Name                            Score  Genres
----  ---------------------------  --------  ---------------------------------------------
 595  Beauty and the Beast (1991)  0.907676  Animation|Children's|Musical
2294  Antz (1998)                  0.897919  Animation|Children's
2687  Tarzan (1999)                0.896598  Animation|Children's
 364  Lion King, The (1994)        0.893712  Animation|Children's|Musical
1566  Hercules (1997)              0.891306  Adventure|Animation|Children's|Comedy|Musical
1907  Mulan (1998)                 0.883115  Animation|Children's
   1  Toy Story (1995)             0.881465  Animation|Children's|Comedy
   8  Tom and Huck (1995)          0.875803  Adventure|Children's
3114  Toy Story 2 (1999)           0.875504  Animation|Children's|Comedy
'''

from __future__ import print_function
import os
import sys
import time
import theano
import csv
import numpy as np
import operator
import scipy
import gc
from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, Flatten, merge
from collections import Counter, deque
from random import random as random01
import random
from tabulate import tabulate

def LoadMovieLensData(movies_filepath, rating_filepath, positive_score_threshold = 3.5):
    def UpdateUser2Items(user2items, user, item):
        if user in user2items:
            user2items[user].append(item)
        else:
            user2items[user] = [item]
    item2genre = {}
    item2name = {}
    user2items = {}
    # Parse movieId,title,genres        
    with open(movies_filepath) as reader:
        for line in reader:
            line = line.split("::")
            item2genre[line[0]] = line[2]
            item2name[line[0]] = line[1]

    # Parse userId,movieId,rating,timestamp
    with open(rating_filepath) as reader:
        for line in reader:
            line = line.split("::")
            if float(line[2]) > positive_score_threshold:
                UpdateUser2Items(user2items, line[0], line[1])
    return user2items, item2name, item2genre
        
def GetEmbeddingLayer(num_items, embedding_dim, l2_reg = None):
    return Embedding(num_items, embedding_dim) if l2_reg == None else Embedding(num_items, embedding_dim, W_regularizer = l2(l2_reg))
    

def Item2vecModel(num_items, embedding_dim, l2_reg = None, dual_embedding = True, bias = False):
    input_u = Input(shape = (1,))
    input_v = Input(shape = (1,))
    u = GetEmbeddingLayer(num_items, embedding_dim, l2_reg)
    v = GetEmbeddingLayer(num_items, embedding_dim, l2_reg) if dual_embedding else u
    u = u(input_u)
    v = v(input_v)
    merge_layer = merge([u, v], mode = 'dot')
    if bias:
        b_u = GetEmbeddingLayer(num_items, 1)
        b_v = GetEmbeddingLayer(num_items, 1) if dual_embedding else b_u
        b_u = b_u(input_u)
        b_v = b_v(input_v)
        merge_layer = merge([merge_layer, b_u, b_v], mode = 'sum')
    merge_layer = Flatten()(merge_layer)
    output_layer = Dense(1, activation = 'sigmoid', bias = False)(merge_layer)
    return Model(input = [input_u, input_v], output = [output_layer])

def FilterItems(item_lists, item_counter, min_count = 3, max_len_list = 10):
    # filter items and uninformative lists
    item_list_filtered = []
    for item_list in item_lists:
        item_list = [item for item in item_list if item_counter[item] > min_count]
        if len(item_list) > 1 and len(item_list) <= max_len_list:
            item_list_filtered.append(item_list)
    return item_list_filtered

def Items2Ints(item_counter):
    # Map items to integers (according to frequency)
    item_counter_sorted = sorted(item_counter.items(), key = operator.itemgetter(1), reverse = True)
    item2int = {}
    int2item = {}
    for i, item_count in enumerate(item_counter_sorted):
        item2int[item_count[0]] = i
        int2item[i] = item_count[0]
    return item2int, int2item

def GetItemCounter(item_lists):
    item_counter = Counter()
    for item_list in item_lists:
            item_counter.update(item_list)
    return item_counter

def GenerateNegativePool(item_counter, total_length, p = 0.75):
    negative_pool = []
    for item, count in item_counter.items():
        negative_pool.extend([item] * int(round(((1.0 * count / total_length) ** p) * total_length)))
    return negative_pool

def SubsamplingTable(item_counter, total_length, rho = 1e-5):
    probs = np.zeros(len(item_counter))
    rho_count = rho * total_length
    for item, count in item_counter.items():
        rho_div_item = count / rho_count
        probs[item] = 1 - (1 + np.sqrt(rho_div_item)) / rho_div_item
         
    return probs

def Skipgrams(sequence, window_size = 4, negative_samples = 1,
              shuffle = True, sampling_table = None, negative_pool = None):
    couples = deque()
    labels = deque()
    pool_length = len(negative_pool)
    range_negative_samples = range(negative_samples)
    for i, wi in enumerate(sequence):
        if not wi:
            continue
        if sampling_table is not None and sampling_table[wi] > random01():
            continue
        window_start = max(0, i - window_size)
        window_end = min(len(sequence), i + window_size + 1)
        for j in xrange(window_start, window_end):
            if j != i:
                wj = sequence[j]
                if not wj:
                    continue
                couples.append([wi, wj])
                labels.append(1)
                negative_indices = [int(random01() * pool_length) for n in range_negative_samples]
                for n in range_negative_samples:
                    couples.append([wi, negative_pool[negative_indices[n]]])    
                    labels.append(0)
    if shuffle:
        seed = random.randint(0, 10e6)
        random.seed(seed)
        random.shuffle(couples)
        random.seed(seed)
        random.shuffle(labels)
    return couples, labels

def GenerateSkipgrams(item_lists, num_items, negative_pool, negative_samples = 1.0,
                      sampling_table = None, window_size = 1000):
    skipgram_data = deque()
    for item_list in item_lists:
        skipgram_data.append(Skipgrams(item_list, negative_pool = negative_pool,
                             window_size = window_size, negative_samples = negative_samples,
                             shuffle = False, sampling_table = sampling_table))
    return skipgram_data

def ForamtDataset(skipgram_data):
    X0 = deque()
    X1 = deque()
    Y = deque()
    for record in skipgram_data:
        for x, y in zip(record[0], record[1]):
            X0.append(x[0])
            X1.append(x[1])
            Y.append(y)
    return [np.array(X0), np.array(X1)], np.array(Y)

def GetMostSimilar(item_vector, item_vectors, top_k = 10, distance_type = 'cosine'):
    dists = scipy.spatial.distance.cdist(item_vector[np.newaxis], item_vectors, distance_type)[0]
    return np.sort(dists)[:top_k], np.argsort(dists)[:top_k]

def PrintMostSimilar(item_id, item_vectors, int2item, item2name, item2genre, top_k = 10):
    dists_sorted, dists_argsorted = GetMostSimilar(item_vectors[item2int[item_id]], item_vectors, top_k = top_k + 1)
    print(" ".join(["Seed: ", item_id, item2name[item_id], item2genre[item_id]]))
    records = [[int2item[j], ToAscii(item2name[int2item[j]]), 1 - s, item2genre[int2item[j]]] for j, s in zip(dists_argsorted[1:],
                                                                                                              dists_sorted[1:])]
    print (tabulate(records, ["ID", "Name", "Score", "Genres"]))

def SampleDataset(item_lists, num_items, negative_pool, negative_samples, sampling_table, window_size):
    gc.disable()
    t = time.clock()
    print("Generating skipgrams ...", end = "")
    sys.stdout.flush()
    skipgram_data = GenerateSkipgrams(item_lists, num_items, negative_pool, negative_samples, sampling_table, window_size)
    random.shuffle(skipgram_data)
    print("{:.2f}s".format(time.clock() - t), end = "  ")
    t = time.clock()
    print("Formatting dataset ...", end = "")
    sys.stdout.flush()
    X, Y = ForamtDataset(skipgram_data)
    print("{:.2f}s".format(time.clock() - t))
    sys.stdout.flush()
    gc.enable()
    return X, Y
    
random.seed(1337)
dataset_dirpath = ''
movies_filepath = os.path.join(dataset_dirpath, 'movies.dat')
ratings_filepath = os.path.join(dataset_dirpath, 'ratings.dat')
# Filter all items with count < min_count
min_count = 20
# Filter all lists with size > max_len_list
max_len_list = 100
num_epoch = 25
batch_size = 8192
# Resample dataset between successive epochs
resample = True
# Sampling factor (rho in the paper)
sampling_factor = 1e-4
# Window size
window_size = 10000
# Target embedding dimension
embedding_dim = 40
# Number of negative samples
negative_samples = 10
# Power to raise the unigram distribtion
p = 0.75
# L2 regularization parameter
l2_reg = None
# Set to False to use a single set of embedding
dual_embedding = True
# Set to True to use biases
bias = False

# Read movielens data and produce lists of co-occurence items
user2items, item2name, item2genre = LoadMovieLensData(movies_filepath, ratings_filepath, positive_score_threshold = 3.5)
item_lists_orig = user2items.values()
item_counter = GetItemCounter(item_lists_orig)

# Filter items with count < min_count, lists with len > max_len_list and unimformative lists (len < 2)
item_lists = FilterItems(item_lists_orig, item_counter, min_count, max_len_list)
random.shuffle(item_lists)
item_counter = GetItemCounter(item_lists)

# Map items to integers
item2int, int2item = Items2Ints(item_counter)
num_items = len(item2int)
item_lists = [[item2int[item] for item in item_list] for item_list in item_lists]
total_length = sum([len(item_list) for item_list in item_lists])
item_counter = GetItemCounter(item_lists)

# Create subsampling table and negative pool (for negative sampling)
sampling_table = SubsamplingTable(item_counter, total_length, rho = sampling_factor)
negative_pool = GenerateNegativePool(item_counter, total_length, p)

# Generate dataset
X, Y = SampleDataset(item_lists, num_items, negative_pool, negative_samples, sampling_table, window_size)
model = Item2vecModel(num_items, embedding_dim, l2_reg, dual_embedding, bias)
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print("Training item2vec model on {} unique items and {} lists".format(num_items, len(item_lists)))
sys.stdout.flush()
for i in range(num_epoch):
    print("Epoch {}: ".format(i), end = "")
    sys.stdout.flush()
    info = model.fit(X, Y, batch_size = batch_size, nb_epoch = 1, shuffle = False, verbose = 2)
    if resample:
        X, Y = SampleDataset(item_lists, num_items, negative_pool, negative_samples, sampling_table, window_size)

# Get embeddding
U = model.layers[2].get_weights()[0]
V = model.layers[3].get_weights()[0]

# Print results for movie ID 588"
PrintMostSimilar("588", U, int2item, item2name, item2genre, top_k = 10)
