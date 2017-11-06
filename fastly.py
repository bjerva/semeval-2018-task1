#import utils
#import data_utils
#from analysis import write_confusion_matrix, prepare_error_analysis
#from config import *
import numpy as np

#word_vec_map, word_id_map, vec_dim = utils.read_word_embeddings("data/2018-EI-oc-Enanger-dev.txt")

#print(type(word_vec_map))
#print(type(word_id_map))
#print(vec_dim)

word_vec_map = {}
word_id_map = {}
with open("en.polyglot.txt.dummy", 'r', encoding='utf-8') as in_f:
    for idx, line in enumerate(in_f):
        fields = line.strip().split()
        word = fields[0]
        embedding = np.asarray([float(i) for i in fields[1:]], dtype=np.float32)

        word_vec_map[word] = embedding
        word_id_map[word] = len(word_id_map)

# get dimensions from last word added
vec_dim = len(word_vec_map[word])
print(word_vec_map["United"])
print(word_id_map["United"])
print(vec_dim)

