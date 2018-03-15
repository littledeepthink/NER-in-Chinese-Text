# coding = utf-8
import numpy as np
import os
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

embedding_file = 'data/char_embedding_matrix.npy'

def get_char_tag_data(file_path):
    with open(file_path, 'r') as f:
        list_all = f.readlines() # type: list
    # print(list_all, len(list_all))
    # ['本 O\n', '性 O\n', '的 O\n', '差 O\n', '别 O\n', '。 O\n', '\n'] 112188
    i = 0
    char_str = str()
    char_list = [] # 列表中每个元素为每句话组成的字符串
    tag_str = str()
    tag_list = [] # 列表中的每个元素为每句话对应的tag所组成的字符串
    while i < len(list_all)-1:
        str_all = list_all[i]
        # print(str_all)
        tep_list = str_all.split(' ')
        if (len(tep_list) > 1) & (tep_list[0] not in '!。?;'):
            char_str += (tep_list[0] + ' ')
            tag_str += tep_list[1]
        else:
            if tep_list[0] in '!。?;':
                char_str += (tep_list[0] + ' ')
                tag_str += tep_list[1]
            char_list.append(char_str)
            tag_list.append(tag_str)
            char_str = str()
            tag_str = str()
        i += 1
    # print(char_list[:3], tag_list[:3])

    char_data = [sent.split() for sent in char_list if len(sent) > 0] # 将每句话转化为由单字符字符串构成的列表
    tag_data = [tags.split('\n')[:-1] for tags in tag_list if len(tags) > 0] # 同上, 专门去掉''
    # 'O\nLOC\nO\n'.split('\n') : ['O', 'LOC', 'O', '']    !!!!
    return char_data, tag_data

def get_char2object():
    char2vec = {}
    f = open('wiki_100.utf8.txt') # load pre-trained word embedding
    i = 0
    for line in f:
        tep_list = line.split()
        if i == 0:
            n_char = int(tep_list[0])
            n_embed = int(tep_list[1])
        else:
            char = tep_list[0]
            vec = np.asarray(tep_list[1:], dtype='float32')
            char2vec[char] = vec
        i += 1
    f.close()
    char2index = {k: i for i, k in enumerate(sorted(char2vec.keys()), 1)}
    return char2vec, n_char, n_embed, char2index

def get_embedding_matrix(char2vec, n_vocab, n_embed, char2index):
    embedding_mat = np.zeros([n_vocab, n_embed])
    for w, i in char2index.items():
        vec = char2vec.get(w)
        if vec is not None:
            embedding_mat[i] = vec
    if not os.path.exists(embedding_file):
        np.save(embedding_file, embedding_mat)
    return embedding_mat

def get_X_data(char_data, char2index, max_length):
    index_data = []
    for l in char_data:
        index_data.append([char2index[s] if char2index.get(s) is not None else 0
                           for s in l])
    index_array = pad_sequences(index_data, maxlen=max_length, dtype='int32',
                                padding='post', truncating='post', value=0)
    return index_array

def get_y_data(tag_data, label2index, max_length):
    index_data = []
    for l in tag_data:
        index_data.append([label2index[s] for s in l])
    index_array = pad_sequences(index_data, maxlen=max_length, dtype='int32',
                                padding='post', truncating='post', value=0)
    index_array = to_categorical(index_array, num_classes=7) # (20863, 574, 7)

    # return np.expand_dims(index_array, -1)
    return index_array

if __name__ == '__main__':

    char_train, tag_train = get_char_tag_data('data/train.txt')
    char_dev, tag_dev = get_char_tag_data('data/dev.txt')
    char_test, tag_test = get_char_tag_data('data/test.txt')
    # print(char_train[:3], tag_train[:3])
    char2vec, n_char, n_embed, char2index = get_char2object()
    n_vocab = n_char + 1
    # print(word2vec['的'], word2index['的']) # n_embed = 100
    if os.path.exists(embedding_file):
        embedding_mat = np.load(embedding_file)
    else:
        embedding_mat = get_embedding_matrix(char2vec, n_vocab, n_embed, char2index)

    # length = []
    # for data in [char_train, char_dev]:
    #     for l in data:
    #         length.append(len(l))
    # print(max(length), length[800:1000]) # 574
    # count = 0
    # for k in length:
    #     if k > 200:
    #         count += 1
    # print(count, len(length)) # 69 23509

    X_train = get_X_data(char_train, char2index, 200)
    X_dev = get_X_data(char_dev, char2index, 200)
    X_test = get_X_data(char_test, char2index, 200)
    print(X_train.shape, X_dev.shape, X_test.shape) # (21147, 200) (2362, 200) (4706, 200)

    # tag_set = set()
    # for data in [tag_train, tag_dev, tag_test]:
    #     for l in data:
    #         tag_set.update(l)
    # print(tag_set) # {'B-LOC', 'B-PER', 'I-LOC', 'I-ORG', 'I-PER', 'O', 'B-ORG'}
    label2index = dict()
    idx = 0
    for c in ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']:
        label2index[c] = idx
        idx += 1
    # print(label2index)

    y_train = get_y_data(tag_train, label2index, 200)
    y_dev = get_y_data(tag_dev, label2index, 200)
    y_test = get_y_data(tag_test, label2index, 200)
    # print(y_train[:2])

    np.save('data/X_train.npy', X_train)
    np.save('data/X_dev.npy', X_dev)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_dev.npy', y_dev)
    np.save('data/y_test.npy', y_test)


# import re
# data = open('data/dev.txt').read()
# lst = re.split(r'[？。!]', data.replace('\n', ' ').replace(' ', ''))

# rep = {' ': '', '\n': ''} # 多重同时替换
# dt = dict((re.escape(k), v) for k, v in rep.items())
# pattern = re.compile("|".join(dt.keys()))
# my_str = pattern.sub(lambda m: dt[re.escape(m.group(0))], data)
#
# lst = re.split(r'[？。!]', my_str)
# print(lst[:4])

# list_all = [lst[0]]
# list_all.extend( [lst[i][1:] for i in range(1, len(lst))] )
# print(list_all[:4])

# Remark:
#>>> '主 O\n'.split()
# ['主', 'O']
# >>> '主 O\n'.split(' ')
# ['主', 'O\n']
