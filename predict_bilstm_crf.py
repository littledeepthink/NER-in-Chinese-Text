# coding=utf-8
import numpy as np
from bilstm_crf import BiLSTM_CRF
from collections import defaultdict
import preprocess as p

def get_X_orig(X_data, index2char):
    """
    :param X_data: index_array
    :param index2char: dict
    :return: 以character_level text列表为元素的列表
    """
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
    """
    :param X_data: 以character_level text列表为元素的列表
    :param y_data: 以entity列表为元素的列表
    :return: [{'entity': [phrase or word], ....}, ...]
    """
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
                d[l[2:]][-1] += c
            elif l == 'O':
                entity_name = ''
        entity_list.append(d)

    return entity_list

def micro_evaluation(pred_entity, true_entity):
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

    X_test = np.load('data/X_test.npy')
    # print(X_test, X_test.shape)
    y_test = np.load('data/y_test.npy')

    ## loss: 16.68228; 
    ## micro: precision:0.7741 recall:0.7641 F1:0.7691 LSTM
    # ner_model = BiLSTM_CRF(n_input=200, n_vocab=char_embedding_mat.shape[0],
    #                        n_embed=100, embedding_mat=char_embedding_mat,
    #                        keep_prob=0.5, n_lstm=128, keep_prob_lstm=0.7,
    #                        n_entity=7, optimizer='adam', batch_size=32, epochs=500)

    #loss: 16.68265;
    # micro: precision:0.7798; recall:0.7625; F1:0.771  GRU
    # macro: 0.7914 0.7677 0.7791
    ner_model = BiLSTM_CRF(n_input=200, n_vocab=char_embedding_mat.shape[0],
                           n_embed=100, embedding_mat=char_embedding_mat,
                           keep_prob=0.5, n_lstm=256, keep_prob_lstm=0.6,
                           n_entity=7, optimizer='adam', batch_size=16, epochs=500)

    model_file = 'checkpoints/bilstm_crf_best.hdf5'
    ner_model.model.load_weights(model_file)

    y_pred = ner_model.model.predict(X_test[:, :])
    # print(pred.shape) # (4635, 574, 7)

    char2vec, n_char, n_embed, char2index = p.get_char2object()
    # print(word2index['我'])
    index2char = {i: w for w, i in char2index.items()}
    # print(index2word[6593])
    X_list = get_X_orig(X_test[:, :], index2char) # list

    pred_list, true_list = get_y_orig(y_pred, y_test[:, :]) # list
    # print(X_list)
    pred_entity, true_entity = get_entity(X_list, pred_list), get_entity(X_list, true_list)
    # print(pred_entity, true_entity)
    precision, recall, f1 = micro_evaluation(pred_entity, true_entity)
    print(precision, recall, f1)
    
    # Just test 'get_entity' function:
    # X_true = [['火','箭','队','的','主','场','在','休','斯','顿',',',
    #            '当','家','球','星','为','哈','登','和','保','罗'],
    #           ['北', '京', '故', '宫', '主','场','在','休','斯','顿',',',
    #            '当','家','球','星','为','哈','登','和','保','罗']]
    # pred_list = [['B-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'B-LOC',
    #               'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER',
    #               'I-PER', 'O', 'B-PER', 'I-PER'],['B-LOC', 'I-LOC', 'B-ORG',
    #                 'I-ORG', 'O', 'O', 'O', 'B-LOC',
    #                 'I-LOC', 'I-LOC', 'O', 'O', 'O', 'O', 'O', 'O', 'B-PER',
    #                 'I-PER', 'O', 'B-PER', 'I-PER']]
    # entity_list = get_entity(X_true, pred_list)
    # print(entity_list)
