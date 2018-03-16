# coding=utf-8
from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout,\
                         TimeDistributed, Concatenate, Dense, GRU, Conv1D,\
                         LeakyReLU
from keras.layers.normalization import BatchNormalization
from crf_layer import CRF

class BiLSTM_CRF():
    """
    两输入：main_input(字序列), auxiliary_input(对应的相同timestep的词序列)
    单输出：char_wise IOB label
    """
    def __init__(self, n_input_char, char_embedding_mat, n_input_word,
                  keep_prob, n_lstm, keep_prob_lstm, n_entity,
                  optimizer, batch_size, epochs, word_embedding_mat=None,
                  n_filter=None, kernel_size=None,):
        self.n_input_char = n_input_char
        self.char_embedding_mat = char_embedding_mat
        self.n_vocab_char = char_embedding_mat.shape[0]
        self.n_embed_char = char_embedding_mat.shape[1]
        self.n_input_word = n_input_word
        self.word_embedding_mat = word_embedding_mat
        self.n_vocab_word = word_embedding_mat.shape[0]
        self.n_embed_word = word_embedding_mat.shape[1]
        self.keep_prob = keep_prob
        self.n_lstm = n_lstm
        self.keep_prob_lstm = keep_prob_lstm
        self.n_filter = n_filter
        self.kernel_size = kernel_size

        self.n_entity = n_entity
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs

        self.build()
        self.build2()

    def build(self):
        # main
        char_input = Input(shape=(self.n_input_char,), name='main_input')
        char_embed = Embedding(input_dim=self.n_vocab_char,
                          output_dim=self.n_embed_char,
                          weights=[self.char_embedding_mat],
                          input_length=self.n_input_char,
                          mask_zero=True,
                          trainable=True)(char_input)
        char_embed_drop = Dropout(self.keep_prob)(char_embed)
        bilstm = Bidirectional(GRU(self.n_lstm, return_sequences=True,
                                    dropout=self.keep_prob_lstm,
                                    recurrent_dropout=self.keep_prob_lstm)
                               )(char_embed_drop)
        # auxiliary
        word_input = Input(shape=(self.n_input_word,), name='auxiliary_input')
        word_embed = Embedding(input_dim=self.n_vocab_word,
                               output_dim=self.n_embed_word,
                               weights=[self.word_embedding_mat],
                               input_length=self.n_input_word,
                               mask_zero=True,
                               trainable=True)(word_input)
        word_embed_drop = Dropout(self.keep_prob)(word_embed)
        lstm = Bidirectional(GRU(self.n_lstm, return_sequences=True,
                                  dropout=self.keep_prob_lstm,
                                  recurrent_dropout=self.keep_prob_lstm)
                             )(word_embed_drop)

        # concatenation
        concat = Concatenate(axis=-1)([bilstm, lstm])
        concat_drop = TimeDistributed(Dropout(self.keep_prob))(concat)

        crf = CRF(units=self.n_entity, learn_mode='join',
                  test_mode='viterbi', sparse_target=False)
        output = crf(concat_drop)

        self.model = Model(inputs=[char_input, word_input],
                           outputs=output)
        self.model.compile(optimizer=self.optimizer,
                           loss=crf.loss_function,
                           metrics=[crf.accuracy])

    def build2(self):
        # main
        char_input = Input(shape=(self.n_input_char,))
        char_embed = Embedding(input_dim=self.n_vocab_char,
                               output_dim=self.n_embed_char,
                               input_length=self.n_input_char,
                               weights=[self.char_embedding_mat],
                               mask_zero=False,
                               trainable=True)(char_input)
        char_embed_drop = Dropout(self.keep_prob)(char_embed)
        # auxiliary
        word_input = Input(shape=(self.n_input_word,))
        word_embed = Embedding(input_dim=self.n_vocab_word,
                               output_dim=self.n_embed_word,
                               input_length=self.n_input_word,
                               weights=[self.word_embedding_mat],
                               mask_zero=False,
                               trainable=True)(word_input)
        word_embed_drop = Dropout(self.keep_prob)(word_embed)
        # 使用CNN提取word的n_gram特征
        word_conv = Conv1D(self.n_filter, kernel_size=self.kernel_size,
                           strides=1, padding='same',
                           kernel_initializer='he_normal')(word_embed_drop)
        word_conv = BatchNormalization(axis=-1)(word_conv)
        word_conv = LeakyReLU(alpha=1/5.5)(word_conv)
　　　　　# concatenation
        concat = Concatenate(axis=-1)([char_embed, word_conv])
        concat_drop = TimeDistributed(Dropout(self.keep_prob))(concat)

        bilstm = Bidirectional(LSTM(units=self.n_lstm,
                                    return_sequences=True,
                                    dropout=self.keep_prob_lstm,
                                    recurrent_dropout=self.keep_prob_lstm)
                               )(concat_drop)

        crf = CRF(units=self.n_entity, learn_mode='join',
                  test_mode='viterbi', sparse_target=False)
        output = crf(bilstm)

        self.model2 = Model(inputs=[char_input, word_input],
                           outputs=output)
        self.model2.compile(optimizer=self.optimizer,
                           loss=crf.loss_function,
                           metrics=[crf.accuracy])

    def train(self, X_train, y_train, X_dev, y_dev, cb):
        self.model.fit(X_train, y_train, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(X_dev, y_dev),
                       callbacks=cb)

    def train2(self, X_train, y_train, X_dev, y_dev, cb):
        self.model2.fit(X_train, y_train, batch_size=self.batch_size,
                        epochs=self.epochs, validation_data=(X_dev, y_dev),
                        callbacks=cb)
