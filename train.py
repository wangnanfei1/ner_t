
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras_contrib.layers import CRF
import pickle
from keras.layers import Embedding ,Bidirectional,LSTM


class Data_set:
    '''
    数据预处理
    '''
    def __init__(self, data_path, label_path, labels):
        with open(data_path, "r",encoding='utf-8') as f:
            self.data = f.read()
        with open(label_path ,'r',encoding='utf-8') as f:
            self.label_data = f.read()
        self.process_data = self.process_data()
        self.label = self.label_process()
        print(self.label[:2])
        self.labels = labels

    def process_data(self):
        train_data = self.data.split("\n\n")
        train_data = [token.split(" ") for token in train_data]
        #train_data.pop()
        return train_data

    def label_process(self):
        train_data = self.label_data.split("\n")
        train_data = [token.split(" ") for token in train_data]
        #train_data.pop()
        return train_data

    def save_vocab(self, save_path):
        all_char = [char for sen in self.process_data for char in sen]
        chars = set(all_char)
        word2id = {char: id_ + 1 for id_, char in enumerate(chars)}
        word2id["unk"] = 0
        with open(save_path, "wb") as f:
            pickle.dump(word2id, f)
        return word2id

    def generate_data(self, vocab, maxlen):
        char_data_sen = self.process_data
        label_sen =  self.label
        sen2id = [[vocab.get(char, 0) for char in sen] for sen in char_data_sen]
        label2id = {label: id_ for id_, label in enumerate(self.labels)}
        lab_sen2id = [[label2id.get(lab, 0) for lab in sen] for sen in label_sen]
        sen_pad = pad_sequences(sen2id, maxlen)
        lab_pad = pad_sequences(lab_sen2id, maxlen, value=-1)
        print(lab_pad[:2])
        lab_pad = np.expand_dims(lab_pad, 2)
        return sen_pad, lab_pad


class Ner:
    def __init__(self, vocab, labels_category, Embedding_dim=200):
        self.Embedding_dim = Embedding_dim
        self.vocab = vocab
        self.labels_category = labels_category
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Embedding(len(self.vocab), self.Embedding_dim, mask_zero=True))  # Random embedding
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        crf = CRF(len(self.labels_category), sparse_target=True)
        model.add(crf)
        model.summary()
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    def train(self, data, label, EPOCHS):
        self.model.fit(data, label, batch_size=16, epochs=EPOCHS)
        self.model.save('crf.h5')

    def predict(self, model_path, data, maxlen):
        model = self.model
        char2id = [self.vocab.get(i) for i in data]
        pad_num = maxlen - len(char2id)
        input_data = pad_sequences([char2id], maxlen)
        model.load_weights(model_path)
        result = model.predict(input_data)[0][-len(data):]
        result_label = [np.argmax(i) for i in result]
        return result_label

if __name__ == "__main__":
    # import os
    # print (os.getcwd())
    # data_path = r'./train.txt'
    # with open(data_path, 'r', encoding='utf-8') as f:
    #     data = f.read()
    # train_data = data.split("\n\n")
    # train_data = [token.split() for token in train_data]
    # print(train_data[:10])
    data_path =  r'./train.txt'
    label_path = r'./label.txt'
    data = Data_set(data_path,label_path,['O', 'F', 'A'])
    vocab = data.save_vocab("vocab.pk")
    sentence, sen_tags = data.generate_data(vocab, 20)
    print(sentence[:2], sen_tags[:2])

    tags = ['O', 'F', 'A']
    ner = Ner(vocab, tags)

    ner.train(sentence, sen_tags,1)