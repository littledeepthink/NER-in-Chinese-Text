# NER:Chinese named entity recognition in Keras

Model1：Character_embedding-Based BiLSTM-CRF. 
Model2：On the basis of model1. Extract n_gram feature from word embedding as auxiliary feature, using Conv1D.  

注：先运行preprocess.py和utils.py以生成train、dev和test。

References:
1.End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. Xuezhe Ma, Eduard Hovy
2.Bidirectional LSTM-CRF Models for Sequence Tagging. Zhiheng Huang, Wei Xu, Kai Yu.
3.Neural Architectures for Named Entity Recognition. Guillaume Lample et al.
