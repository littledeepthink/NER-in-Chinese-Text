# coding=utf-8
import numpy as np
from bilstm_crf_add_word import BiLSTM_CRF
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,\
                            TensorBoard
from keras.optimizers import Adam, Nadam
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

char_embedding_mat = np.load('data/char_embedding_matrix.npy')
# word_embedding_mat = np.load('data/word_embedding_matrix.npy')
word_embedding_mat = np.random.randn(157142, 200)

X_train = np.load('data/X_train.npy')
train_add = np.load('data/word_train_add.npy') # add word_embedding
X_dev = np.load('data/X_dev.npy')
dev_add = np.load('data/word_dev_add.npy')
y_train = np.load('data/y_train.npy')
y_dev = np.load('data/y_dev.npy')

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=0.01)
# nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

# ner_model = BiLSTM_CRF(n_input_char=200, char_embedding_mat=char_embedding_mat,
#                        n_input_word=200, word_embedding_mat=word_embedding_mat,
#                        keep_prob=0.7, n_lstm=256, keep_prob_lstm=0.6, n_entity=7,
#                        optimizer=adam, batch_size=32, epochs=500)
ner_model = BiLSTM_CRF(n_input_char=200, char_embedding_mat=char_embedding_mat,
                       n_input_word=200, word_embedding_mat=word_embedding_mat,
                       keep_prob=0.7, n_lstm=256, keep_prob_lstm=0.6, n_entity=7,
                       optimizer=adam, batch_size=32, epochs=500,
                       n_filter=128, kernel_size=3)
cp_folder, cp_file = 'checkpoints', 'bilstm_crf_add_word_weights_best.hdf5'
log_filepath = os.getcwd() + '/logs/bilstm_crf_add_word_summaries'

cb = [ModelCheckpoint(os.path.join(cp_folder, cp_file), monitor='val_loss',
                      verbose=1, save_best_only=True, save_weights_only=True, mode='min'),
      EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=10, mode='min'),
      TensorBoard(log_dir=log_filepath, write_graph=True, write_images=True,
                  histogram_freq=0),
      ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, mode='min',
                        epsilon=1e-4, cooldown=2, min_lr=1e-8)]

ner_model.train2([X_train, train_add], y_train, [X_dev, dev_add], y_dev, cb)

