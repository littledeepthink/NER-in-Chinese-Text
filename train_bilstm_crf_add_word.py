# coding = utf-8
import numpy as np
from bilstm_crf_add_word import BiLSTM_CRF
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

char_embedding_mat = np.load('data/char_embedding_matrix.npy')
word_embedding_mat = np.load('data/word_embedding_matrix.npy')

X_train = np.load('data/X_train.npy')
train_add = np.load('data/word_train_add.npy') # add word_embedding
X_dev = np.load('data/X_dev.npy')
dev_add = np.load('data/word_dev_add.npy')
y_train = np.load('data/y_train.npy')
y_dev = np.load('data/y_dev.npy')

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=0.01)
# 随机初始化word_embedding_mat, n_embed_word=100
ner_model = BiLSTM_CRF(n_input_char=200, char_embedding_mat=char_embedding_mat,
                       n_input_word=200, n_vocab_word=157142, n_embed_word=100,
                       keep_prob=0.5, n_lstm=100, keep_prob_lstm=0.8, n_entity=7,
                       optimizer=adam, batch_size=64, epochs=500,
                       word_embedding_mat=None)
cp_folder, cp_file = 'checkpoints', 'bilstm_crf_add_word_weights_best.hdf5'

cb = [ModelCheckpoint(os.path.join(cp_folder, cp_file), monitor='val_loss',
                      verbose=1, save_best_only=True, save_weights_only=True, mode='min'),
      EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=10, mode='min'),
      ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, verbose=0, mode='min',
                        epsilon=1e-6, cooldown=4, min_lr=1e-8)]

ner_model.train([X_train, train_add], y_train, [X_dev, dev_add], y_dev, cb)
