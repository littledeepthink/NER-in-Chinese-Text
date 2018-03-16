# coding: utf-8
import numpy as np
from bilstm_crf_add_word import BiLSTM_CRF
from collections import defaultdict
import preprocess as p
from keras.optimizers import Adam, Nadam

def get_X_orig(X_data, index2char):
    X_orig = []
    for n in range(X_data.shape[0]):
        orig = [index2char[i] if i > 0 else 'None' for i in X_data[n]]
        X_orig.append(orig)
    return X_orig

def get_y_orig(y_pred, y_true):
    label = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
    index2label = dict()
    idx = 0
    for c in label:
        index2label[idx] = c
        idx += 1
    n_sample = y_pred.shape[0]
    pred_list = []
    true_list = []
    for i in range(n_sample):
        pred_label = [index2label[idx] for idx in np.argmax(y_pred[i], axis=1)]
        pred_list.append(pred_label)
        true_label = [index2label[idx] for idx in np.argmax(y_true[i], axis=1)]
        true_list.append(true_label)
        # print(pred_label, true_label)
    return pred_list, true_list

def get_entity(X_data, y_data):
    n_example = len(X_data)
    entity_list = []
    entity_name = ''
    for i in range(n_example):
        d = defaultdict(list)
        for c, l in zip(X_data[i], y_data[i]):
            if l[0] == 'B':
                d[l[2:]].append('')
                d[l[2:]][-1] += c
                entity_name += c
            elif (l[0] == 'I') & (len(entity_name) > 0):
                try:
                    d[l[2:]][-1] += c
                except IndexError:
                    d[l[2:]].append(c)
            elif l == 'O':
                entity_name = ''
        entity_list.append(d)

    return entity_list

def micro_evaluation(pred_entity, true_entity):
    # Weight all examples equally,favouring the performance on common classes. 
    n_example = len(pred_entity)
    t_pos, true, pred = [], [], []
    for n in range(n_example):
        et_p = pred_entity[n]
        et_t = true_entity[n]
        print('the prediction is', et_p.items(), '\n',
              'the true is', et_t.items())
        t_pos.extend([len(set(et_p[k]) & set(et_t[k]))
                      for k in (et_p.keys() & et_t.keys())])
        pred.extend([len(v) for v in et_p.values()])
        true.extend([len(v) for v in et_t.values()])

    precision = sum(t_pos) / sum(pred) + 1e-8
    recall = sum(t_pos) / sum(true) + 1e-8
    f1 = 2 / (1 / precision + 1 / recall)

    return round(precision, 4), round(recall, 4), round(f1, 4)

def macro_evaluation(pred_entity, true_entity):
    # Weight all classes equally. 
    label = ['PER', 'ORG', 'LOC']
    n_example = len(pred_entity)
    precision, recall, f1 = [], [], []
    for l in label:
        t_pos, true, pred = [], [], []
        for n in range(n_example):
            et_p = pred_entity[n]
            et_t = true_entity[n]
            print('the prediction is', et_p.items(), '\n',
                  'the true is', et_t.items())
            t_pos.extend([len(set(et_p[l]) & set(et_t[l]))
                          if l in (et_p.keys() & et_t.keys()) else 0])
            true.extend([len(et_t[l]) if l in et_t.keys() else 0])
            pred.extend([len(et_p[l]) if l in et_p.keys() else 0])
        precision.append(sum(t_pos) / sum(pred) + 1e-8)
        recall.append(sum(t_pos) / sum(true) + 1e-8)
        f1.append(2 / (1 / precision[-1] + 1 / recall[-1]))
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    return round(avg_precision, 4), round(avg_recall, 4), round(avg_f1, 4)


if __name__ == '__main__':

    char_embedding_mat = np.load('data/char_embedding_matrix.npy')
    # word_embedding_mat = np.load('data/word_embedding_matrix.npy')
    word_embedding_mat = np.random.randn(157142, 200)

    X_test = np.load('data/X_test.npy')
    test_add = np.load('data/word_test_add.npy') # add word_embedding
    # print(X_test, X_test.shape)
    y_test = np.load('data/y_test.npy')

    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipvalue=0.01)
    # nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    ner_model = BiLSTM_CRF(n_input_char=200, char_embedding_mat=char_embedding_mat,
                           n_input_word=200, word_embedding_mat=word_embedding_mat,
                           keep_prob=0.7, n_lstm=256, keep_prob_lstm=0.6, n_entity=7,
                           optimizer=adam, batch_size=32, epochs=500,
                           n_filter=128, kernel_size=3)
    model_file = 'checkpoints/bilstm_crf_add_word_weights_best.hdf5'
    ner_model.model2.load_weights(model_file)

    y_pred = ner_model.model2.predict([X_test[:, :], test_add[:, :]])
    # print(pred.shape) # (4635, 574, 7)

    char2vec, n_char, n_embed, char2index = p.get_char2object()
    index2char = {i: w for w, i in char2index.items()}

    X_list = get_X_orig(X_test[:, :], index2char) # list

    pred_list, true_list = get_y_orig(y_pred, y_test[:, :]) # list
    # print(X_list)
    pred_entity, true_entity = get_entity(X_list, pred_list), get_entity(X_list, true_list)
    # print(pred_entity, true_entity)
    precision, recall, f1 = macro_evaluation(pred_entity, true_entity)
    print(precision, recall, f1)
