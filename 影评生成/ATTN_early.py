# coding: utf-8

import os
import gensim
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from gensim.models import KeyedVectors
import numpy as np
import torch
from torch import nn, autograd, optim
import torch.nn as nn
import torch.nn.functional as F
# from sru import SRU, SRUCell
import time
import math
from tqdm import tqdm
import collections
import random
import pandas as pd
import matplotlib.pyplot as plt

# ############################ 命名,调参区 ###################################
name = 'att1'

rate_to_train = 1
training_data_name = 'train_set.txt'
testing_data_name = 'test_set.txt'
onehot_dimension = 300


# 加载电影名，并生成movies onehot
def transform_move_to_onehot(movies):
    movie_to_idx = {}
    idx_to_movie = {}
    movie_to_onehot = np.zeros((len(movie_names), len(movie_names)))
    for i, name in enumerate(movies):
        onehot = np.zeros(len(movie_names))
        onehot[i] = 1
        movie_to_idx[name] = i
        movie_to_onehot[i] = onehot
        idx_to_movie[i] = name
    return movie_to_onehot, movie_to_idx, idx_to_movie


movies = pd.read_csv('movie_names_20.csv')
a = movies['movie_names']
print(a)
movie_names = list(movies['movie_names'])
print(movie_names)
movie_to_onehot, movie_to_idx, idx_to_movie = transform_move_to_onehot(movie_names)
movie_size = len(a)

print(movie_to_onehot)

#  词向量加载
fvec = KeyedVectors.load_word2vec_format('vec_100d.txt', binary=False)
word_vec = fvec.vectors
vocab = ['<PAD>', '<BOS>', '<EOS>', '<UNK>']
vocab.extend(list(fvec.vocab.keys()))
word_vec = np.concatenate((np.array([[0] * word_vec.shape[1]] * 4), word_vec))
word_vec = torch.tensor(word_vec).float()

# word_to_idx,idx_to_word
word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}
word_to_label = {'一星级': 0, '二星级': 1, '三星级': 2, '四星级': 3, '五星级': 4}


# 加载训练data,并分成essay,movie,topic
def load_train_data(training_data_name):
    essays = []
    topics = []
    movies = []
    with open(training_data_name, 'r', encoding='utf-8') as f:
        for line in f:
            (essay, topic) = line.replace('\n', '').split(' </d> ')
            words = essay.split(' ')
            new_words = []
            for word in words:
                if word in word_to_idx:
                    new_words.append(word)
                else:
                    for i in range(len(word)):
                        new_words.append(word[i])
            essays.append(new_words)
            topic_tmp = topic.split(' ')
            movies.append(topic_tmp[0])
            topics.append(topic_tmp[1:])

    # references获取, 用于bleu
    a = []
    for i in range(len(movies)):
        a.append((movies[i], topics[i][0]))
    references = {}
    for i, data in enumerate(a):
        if data not in references.keys():
            references[data] = []
            references[data].append(essays[i])
        else:
            references[data].append(essays[i])

    # 每个（电影，主题）取100条评论，不足100的不管,还有去掉测试集数据
    counter = {}
    essays_new = []
    movies_new = []
    topics_new = []
    for i in range(len(movies)):
        if (movies[i], topics[i][0]) in counter.keys():
            counter[(movies[i], topics[i][0])] += 1
        else:
            counter[(movies[i], topics[i][0])] = 1
        if counter[(movies[i], topics[i][0])] <= 100:
            essays_new.append(essays[i])
            movies_new.append(movies[i])
            topics_new.append(topics[i])
        else:
            continue
    return essays_new, movies_new, topics_new, references


essays, movies, topics, references = load_train_data(training_data_name)


# 得到测试集数据
def load_test_data(testing_data_name):
    essays = []
    topics = []
    movies = []
    with open(testing_data_name, 'r', encoding='utf-8') as f:
        for line in f:
            essay, topic = line.replace('\n', '').split(' </d> ')
            words = essay.split(' ')
            new_words = []
            for word in words:
                if word in word_to_idx:
                    new_words.append(word)
                else:
                    for i in range(len(word)):
                        new_words.append(word[i])
            essays.append(new_words)
            topic_tmp = topic.split(' ')
            movies.append(topic_tmp[0])
            topics.append(topic_tmp[1:])
    # references获取, 用于bleu
    a = []
    for i in range(len(movies)):
        a.append((movies[i], topics[i][0]))
    for i, data in enumerate(a):
        if data not in references.keys():
            references[data] = []
            references[data].append(essays[i])
        else:
            references[data].append(essays[i])
    return a


test_data = load_test_data(testing_data_name)


# 转化essay,topic,movies data to idx
def divide_data(essays, movies, topics, rate_to_train):
    num_train = int(len(essays) * rate_to_train)
    corpus_indice = list(map(lambda x: [word_to_idx[w] for w in x if w in word_to_idx], essays[:num_train]))
    topics_indice = list(map(lambda x: [word_to_idx[w] for w in x if w in word_to_idx], topics[:num_train]))
    movies_indice = list(map(lambda x: [movie_to_idx[x]], movies[:num_train]))
    labels = np.array(list(map(lambda x: [word_to_label[w] for w in x], topics[:num_train]))).squeeze()
    return corpus_indice, topics_indice, movies_indice, labels


corpus_indice, topics_indice, movies_indice, labels = divide_data(essays, movies, topics, rate_to_train)
print(labels)

length = list(map(lambda x: len(x), corpus_indice))


def tav_data_iterator(corpus_indice_pre, movies_word_pre, topics_indice_pre, batch_size, num_steps, shuffle=True):
    idx = np.arange(len(corpus_indice_pre))
    np.random.shuffle(idx)
    corpus_indice = [corpus_indice_pre[i] for i in idx]
    movies_word = [movies_word_pre[i] for i in idx]
    topics_indice = [topics_indice_pre[i] for i in idx]
    epoch_size = (len(corpus_indice) + batch_size - 1) // batch_size
    for i in range(epoch_size):
        raw_data = corpus_indice[i * batch_size: (i + 1) * batch_size]
        key_words = topics_indice[i * batch_size: (i + 1) * batch_size]
        movie_names = movies_word[i * batch_size: (i + 1) * batch_size]
        data = np.zeros((len(raw_data), num_steps + 1), dtype=np.int64)
        last_word_indices = []
        for i in range(len(raw_data)):
            doc = raw_data[i]
            tmp = [1]
            tmp.extend(doc)
            tmp.extend([2])
            tmp = np.array(tmp, dtype=np.int64)
            _size = tmp.shape[0]
            last_word_indices.append(_size - 2)
            data[i][:_size] = tmp
        key_words = np.array(key_words, dtype=np.int64)
        x = data[:, 0:num_steps]
        y = data[:, 1:]
        mask = np.float32(x != 0)
        x = torch.tensor(x)
        y = torch.tensor(y)
        mask = torch.tensor(mask)
        movie_names = np.array(movie_names)
        movie_names = torch.tensor(movie_names)
        key_words = torch.tensor(key_words)
        yield (x, y, mask, movie_names, key_words, last_word_indices)


def classfication_data_iterator(corpus_indice_pre, movies_word_pre, labels_pre, batch_size, num_steps, shuffle=True):
    idx = np.arange(len(corpus_indice_pre))
    np.random.shuffle(idx)
    corpus_indice = [corpus_indice_pre[i] for i in idx]
    movies_word = [movies_word_pre[i] for i in idx]
    labels = [labels_pre[i] for i in idx]
    epoch_size = (len(corpus_indice) + batch_size - 1) // batch_size
    for i in range(epoch_size):
        raw_data = corpus_indice[i * batch_size: (i + 1) * batch_size]
        labels_data = labels[i * batch_size: (i + 1) * batch_size]
        movie_names = movies_word[i * batch_size: (i + 1) * batch_size]
        data = np.zeros((len(raw_data), num_steps + 1), dtype=np.int64)
        last_word_indices = []
        for i in range(len(raw_data)):
            doc = raw_data[i]
            tmp = [1]
            tmp.extend(doc)
            tmp.extend([2])
            tmp = np.array(tmp, dtype=np.int64)
            _size = tmp.shape[0]
            last_word_indices.append(_size - 2)
            data[i][:_size] = tmp
        labels_data = np.array(labels_data, dtype=np.int64)
        x = data[:, 0:num_steps]
        x = torch.tensor(x)
        movie_names = np.array(movie_names)
        movie_names = torch.tensor(movie_names)
        labels_data = torch.tensor(labels_data)
        yield (x, movie_names, labels_data, last_word_indices)


class AttnGRU(nn.Module):
    def __init__(self, hidden_size, embedding_size, Usize, movie_size, weight, movie_to_onehot, num_layers, num_labels,
                 dropout=0):
        super(AttnGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.movie_size = movie_size
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.embedding_movie = nn.Linear(self.movie_size, embedding_size)
        self.Usize = Usize
        self.U1 = nn.Linear(self.num_layers * hidden_size, self.Usize)
        self.U2 = nn.Linear(embedding_size, self.Usize)
        self.va = nn.Linear(self.Usize, 1)
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.movie_to_onehot = movie_to_onehot
        self.softmax = nn.Softmax(dim=1)
        self.dropout = dropout
        self.rnn = nn.GRU(input_size=2 * self.embedding_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, bidirectional=False,
                          dropout=self.dropout)
        self.fc1 = nn.Linear(hidden_size, 1024)

    def forward(self, inputs, movie, topic, hidden=None):
        #         if(hidden==None):
        #             hidden = torch.zeros((self.num_layers, inputs.shape[0], self.embedding_size))
        # inputs m*1 , hidden s*m*hidden_size, moive m*1, topic m*1
        #         print('inputs', inputs.shape)
        try:
            hidden == None
        except TypeError:
            pass
        else:
            hidden = torch.zeros((self.num_layers, inputs.shape[0], self.embedding_size)).to(device)
        embeddings = self.embedding(inputs)  # m*1*es
        # print(embeddings.shape)
        topics_embed = self.embedding(topic).reshape(topic.shape[0], -1)  # m*es
        movie = self.movie_to_onehot[movie]  # m*1*ms
        movie = torch.tensor(movie.reshape((-1, movie_size)), dtype=torch.float32).to(device)  # m*ms
        embedding_movie = self.embedding_movie(movie.reshape((movie.shape[0], -1)))  # m*es

        g1 = self.va(torch.tanh(self.U1(hidden.reshape(hidden.shape[1], -1)) + self.U2(embedding_movie)))  # g1 m*1
        g2 = self.va(torch.tanh(self.U1(hidden.reshape(hidden.shape[1], -1)) + self.U2(topics_embed)))  # g2 m*1
        g = torch.cat((g1, g2), 1)  # g m*2
        a = self.softmax(g)  # a m*2
        topics = torch.cat((embedding_movie.unsqueeze(1), topics_embed.unsqueeze(1)), 1)  # topics m*2*es
        average_topic = torch.bmm(g.unsqueeze(1), topics)  # m*1*es
        #         print(embeddings.shape)
        #         print(average_topic.shape)
        inputs = torch.cat((embeddings, average_topic), 2).permute([1, 0, 2])  # 1*m*2es
        self.rnn.flatten_parameters()
        state, hidden = self.rnn(inputs)  # state 1*m*hs, hidden s*m*hs
        output = self.fc1(state.reshape(state.shape[1], -1))  # m*labels
        return output, hidden


writer = SummaryWriter()


def train_AttnGRU(model, optimizer, num_epochs, scheduler=None):
    since = time.time()

    #     best_model_wts = copy.deepcopy(model.state_dict())
    #     best_loss = 0.0
    movies = movie_names
    comments = ['一星级', '二星级', '三星级', '四星级', '五星级']
    logSoftmax = nn.LogSoftmax(dim=1)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        running_bleu = 0

        data = tav_data_iterator(
            corpus_indice, movies_indice, topics_indice, batch_size, max(length) + 1)

        for X, Y, mask, movie, topics, _ in tqdm(data):
            X = X.to(device)
            Y = Y.to(device)
            #             movie = movie.to(device)
            mask = mask.to(device)
            topics = topics.to(device)
            #             print(movie.shape)
            #             print(topics.shape)
            tmp_loss = torch.zeros_like(X, dtype=torch.float32)
            hidden = torch.zeros((model.num_layers, X.shape[0], model.hidden_size)).to(device)
            optimizer.zero_grad()
            for i in range(X.shape[1]):
                with torch.set_grad_enabled(True):
                    output, hidden = model(X[:, i].unsqueeze(1), movie, topics, hidden)
                    # output m*d Y[:,i] m
                    l, _ = adaptive_softmax(output, Y[:, i])
                    tmp_loss[:, i] = -l
            #                 torch.cuda.empty_cache()
            tmp_loss = tmp_loss * mask
            tmp_loss = tmp_loss.sum(dim=1) / mask.sum(dim=1)
            loss = tmp_loss.mean()
            running_loss += loss.item() * X.shape[0]
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / len(corpus_indice)
        print('{} Training loss: {:.4f} '.format(epoch + 1, epoch_loss))
        loss_dict.append(epoch_loss)
        if epoch_loss == min(loss_dict):
            params = model.state_dict()
            torch.save(params, name + ' att_loss_min_modelTAT.pkl')
            torch.save(adaptive_softmax.state_dict(), name + ' att_loss_min_adaptive.pkl')
        writer.add_scalar('loss', epoch_loss, epoch)

        scores_train = []
        scores_test = []
        with open('att_training_process.txt', 'a') as f:
            f.write('\n' + '训练轮数：{}'.format(epoch))
            for movie in movies:
                for comment in comments:
                    (tmp, b) = predict_rnn(model, [movie], [comment], 100, idx_to_word, word_to_idx, movie_to_idx)

                    # bleu
                    if (movie, comment) in references:
                        c = list(map(lambda x: [idx_to_word[int(x)]], b))
                        d = []
                        for data in c:
                            d.append(data[0])
                        candidates = [d]
                        reference = [references[(movie, comment)]]
                        score = evaluation(candidates, reference)
                        f.write('\n' + '{},{},得分是：{}  '.format(movie, comment, score) + tmp)
                        print(movie, comment, ':', tmp, '  bleu得分:', score)
                        scores_train.append(score)
                    else:
                        f.write('\n' + '{},{}：'.format(movie, comment) + tmp)
                        print(movie, comment, ':', tmp)

            print('测试集：')
            f.write('\n测试集：')
            remember = []
            for a in test_data:
                if a in remember:
                    continue
                else:
                    remember.append(a)
                    (tmp, b) = predict_rnn(model, [a[0]], [a[1]], 100, idx_to_word, word_to_idx, movie_to_idx)
                    c = list(map(lambda x: [idx_to_word[int(x)]], b))
                    d = []
                    for data in c:
                        d.append(data[0])
                    candidates = [d]
                    reference = [references[a]]
                    score = evaluation(candidates, reference)
                    f.write('\n' + '{},{},得分是：{}  '.format(a[0], a[1], score) + tmp)
                    print(a[0], a[1], ':', tmp, '  bleu得分:', score)
                    scores_test.append(score)
            f.close()
        scores_train = np.array(scores_train)
        scores_test = np.array(scores_test)
        mean_score_train = scores_train.mean()
        mean_score_test = scores_test.mean()
        train_bleu.append(mean_score_train)
        test_bleu.append(mean_score_test)
        writer.add_scalar('bleu_average_train', mean_score_train, epoch)
        writer.add_scalar('bleu_average_test', mean_score_test, epoch)
        if mean_score_train == max(train_bleu):
            torch.save(model.state_dict(), name + ' train_att_bleu_max_modelTAT.pkl')
            torch.save(adaptive_softmax.state_dict(), name + ' train_att_bleu_max_adaptive.pkl')
        if mean_score_test == max(test_bleu):
            torch.save(model.state_dict(), name + ' test_att_bleu_max_modelTAT.pkl')
            torch.save(adaptive_softmax.state_dict(), name + ' test_att_bleu_max_adaptive.pkl')

        # 早停止
        case = 1
        if len(loss_dict) > 8:
            for i in range(10):
                if loss_dict[-1] <= loss_dict[-2 - i]:
                    case = 0
        if case == 1 and len(loss_dict) > 8:
            break

    return model


def predict_rnn(model, movie, topic, max_num, idx_to_word, word_to_idx, movie_to_idx):
    movie_indice = torch.tensor(movie_to_idx[movie[0]]).view((1, 1)).to(device)
    topic_indice = torch.tensor(word_to_idx[topic[0]]).view((1, 1)).to(device)
    hidden = torch.zeros(model.num_layers, 1, model.hidden_size).to(device)
    output = [1]
    for t in range(max_num):
        X = torch.tensor(output[-1]).reshape((1, 1)).to(device)
        pred, hidden = model(X, movie_indice, topic_indice, hidden)
        pred = adaptive_softmax.predict(pred)
        if pred[-1] == 2:
            break
        else:
            output.append(int(pred[-1]))
    return (''.join([idx_to_word[i] for i in output[1:]]), output[1:])


def evaluation(predicted_comments, original_comments):
    references = original_comments
    candidates = predicted_comments
    score = corpus_bleu(references, candidates)
    return score


device = 'cuda:0'
vocab_size = len(vocab)
hidden_size = 300
embedding_size = 300
Usize = 100
num_layers = 1
dropout = 0.5
adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
    1024, len(vocab), cutoffs=[round(vocab_size / 20), 4 * round(vocab_size / 20)]).to(device)
model_attn = AttnGRU(hidden_size=hidden_size, embedding_size=embedding_size, Usize=Usize, movie_size=movie_size,
                     weight=word_vec,
                     movie_to_onehot=movie_to_onehot, num_layers=num_layers, num_labels=len(vocab),
                     dropout=dropout).to(device)
learning_rate = 1e-3
optimizer = optim.Adam(model_attn.parameters(), lr=learning_rate)
num_epoch = 200
batch_size = 8
use_gpu = True

loss_dict = []
train_bleu = []
test_bleu = []

model_attn = train_AttnGRU(model_attn, optimizer, num_epoch)

torch.save(adaptive_softmax.state_dict(), 'adaptive.pkl')
torch.save(model_attn.state_dict(), 'modelTAT.pkl')

print('\n' + '训练后结果：')
comments = ['一星级', '二星级', '三星级', '四星级', '五星级']
for movie in movie_names:
    for comment in comments:
        (tmp, b) = predict_rnn(model_attn, [movie], [comment], 100, idx_to_word, word_to_idx, movie_to_idx)
        # bleu
        if (movie, comment) in references:
            c = list(map(lambda x: [idx_to_word[int(x)]], b))
            d = []
            for data in c:
                d.append(data[0])
            candidates = [d]
            reference = [references[(movie, comment)]]
            score = evaluation(candidates, reference)
            print(movie, comment, ':', tmp, '  bleu得分:', score)
        else:
            print(movie, comment, ':', tmp)

with open(name + 'att_early_training_process.txt', 'a') as f:
    f.write('\n' + 'review训练过程参数变化')
    f.write('\n' + 'train_bleu:{}'.format(train_bleu))
    f.write('\n' + 'test_bleu:{}'.format(test_bleu))
    f.write('\n' + 'train_loss:{}'.format(loss_dict))


class ClassfiactionAttnGRU(nn.Module):
    def __init__(self, hidden_size, embedding_size, Usize, movie_size, weight, movie_to_onehot, num_layers, num_labels,
                 dropout=0):
        super(ClassfiactionAttnGRU, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.movie_size = movie_size
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.embedding_movie = nn.Linear(self.movie_size, embedding_size)
        self.Usize = Usize
        self.U1 = nn.Linear(self.num_layers * hidden_size, self.Usize)
        self.U2 = nn.Linear(embedding_size, self.Usize)
        self.va = nn.Linear(self.Usize, 1)
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.movie_to_onehot = movie_to_onehot
        self.softmax = nn.Softmax(dim=1)
        self.dropout = dropout
        self.rnn = nn.GRU(input_size=2 * self.embedding_size, hidden_size=self.hidden_size,
                          num_layers=self.num_layers, bidirectional=False,
                          dropout=self.dropout)
        self.fc1 = nn.Linear(hidden_size, 1024)

    def forward(self, inputs, movie, hidden=None):
        #         if(hidden==None):
        #             hidden = torch.zeros((self.num_layers, inputs.shape[0], self.embedding_size))
        # inputs m*1 , hidden s*m*hidden_size, moive m*1, topic m*1
        #         print('inputs', inputs.shape)
        try:
            hidden == None
        except TypeError:
            pass
        else:
            hidden = torch.zeros((self.num_layers, inputs.shape[0], self.embedding_size)).to(device)
        embeddings = self.embedding(inputs)  # m*1*es
        # print(embeddings.shape)
        #         topics_embed = self.embedding(topic).reshape(topic.shape[0], -1) # m*es
        topics_embed = torch.zeros((inputs.shape[0], self.embedding_size), dtype=torch.float32).to(device)
        movie = self.movie_to_onehot[movie]  # m*1*ms
        movie = torch.tensor(movie.reshape((-1, movie_size)), dtype=torch.float32).to(device)  # m*ms
        embedding_movie = self.embedding_movie(movie.reshape((movie.shape[0], -1)))  # m*es

        #         print(hidden.device)
        #         print(embedding_movie.device)
        g1 = self.va(torch.tanh(self.U1(hidden.reshape(hidden.shape[1], -1)) + self.U2(embedding_movie)))  # g1 m*1
        g2 = self.va(torch.tanh(self.U1(hidden.reshape(hidden.shape[1], -1)) + self.U2(topics_embed)))  # g2 m*1
        g = torch.cat((g1, g2), 1)  # g m*2
        a = self.softmax(g)  # a m*2
        topics = torch.cat((embedding_movie.unsqueeze(1), topics_embed.unsqueeze(1)), 1)  # topics m*2*es
        average_topic = torch.bmm(g.unsqueeze(1), topics)  # m*1*es
        #         print(embeddings.shape)
        #         print(average_topic.shape)
        inputs = torch.cat((embeddings, average_topic), 2).permute([1, 0, 2])  # 1*m*2es
        self.rnn.flatten_parameters()
        state, hidden = self.rnn(inputs)  # state 1*m*hs, hidden s*m*hs
        return hidden

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)


device = 'cuda:0'
vocab_size = len(vocab)
hidden_size = 300
embedding_size = 300
Usize = 100
num_layers = 1
dropout = 0.1
model_clf = ClassfiactionAttnGRU(hidden_size=hidden_size, embedding_size=embedding_size, Usize=Usize,
                                 movie_size=movie_size, weight=word_vec,
                                 movie_to_onehot=movie_to_onehot, num_layers=num_layers, num_labels=len(vocab),
                                 dropout=dropout).to(device)
classfication_model = nn.Sequential(
    nn.ReLU(),
    nn.Linear(hidden_size, 512),
    nn.ReLU(),
    nn.Linear(512, 5)
).to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3
optimizer1 = optim.Adam(classfication_model.parameters(), lr=learning_rate)
optimizer2 = optim.Adam(model_clf.parameters(), lr=learning_rate)
num_epoch = 300
batch_size = 1
use_gpu = True

model_clf.load_my_state_dict(model_attn.state_dict())

train_acc = []
test_acc = []
class_loss_dict = []
essays_test, movies_test, topics_test, references = load_train_data(testing_data_name)
corpus_indice_test, topics_indice_test, movies_indice_test, labels_test = divide_data(essays_test, movies_test,
                                                                                      topics_test, rate_to_train)


def train_classficationAttnGRU(model, classfication_model, optimizer1, optimizer2, corpus_indice, movies_indice, labels,
                               batch_size, length):
    for epoch in range(num_epoch):
        start = time.time()
        num, total_loss = 1, 0
        data = classfication_data_iterator(
            corpus_indice, movies_indice, labels, batch_size, max(length) + 1)
        correct = 0
        total = 0
        running_loss = 0.0
        for X, movie, Y, last_word_indices in tqdm(data):

            # X m*L, Y m*L, mask m*L, movie m*1, topics m*1
            X = X.to(device)
            Y = Y.to(device)
            states = torch.zeros((X.shape[1], X.shape[0], model.embedding_size), dtype=torch.float32).to(device)
            hidden = torch.zeros((model.num_layers, X.shape[0], model.hidden_size), dtype=torch.float32).to(device)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            for i in range(X.shape[1]):
                with torch.set_grad_enabled(True):
                    hidden = model(X[:, i].unsqueeze(1), movie, hidden)
                    states[i] = hidden
            states = states.permute([1, 0, 2])
            tmp = torch.zeros((states.shape[0], states.shape[2]), dtype=torch.float32).to(device)
            for i, j in enumerate(last_word_indices):
                tmp[i] = states[i][j]
            pred = classfication_model(tmp)
            predict = torch.argmax(pred, dim=1)
            #             print('predict', predict)
            #             print('Y',Y)
            assert predict.shape == Y.shape
            correct += torch.sum(predict == Y).data
            total += predict.shape[0]
            loss = criterion(pred, Y)
            running_loss += loss.item() * X.shape[0]
            loss.backward()
            optimizer1.step()
            optimizer2.step()
        accuracy = float(correct) / float(total)
        train_acc.append(accuracy)
        class_loss_dict.append(running_loss)
        print('epoch ', epoch, 'accuracy is ', accuracy)
        print('loss', running_loss)
        writer.add_scalar('loss', running_loss, epoch)

        data_test = classfication_data_iterator(
            corpus_indice_test, movies_indice_test, labels_test, batch_size, max(length) + 1)

        correct_test = 0
        total_test = 0
        for X, movie, Y, last_word_indices in tqdm(data_test):
            X = X.to(device)
            Y = Y.to(device)
            states = torch.zeros((X.shape[1], X.shape[0], model.embedding_size), dtype=torch.float32).to(device)
            hidden = torch.zeros((model.num_layers, X.shape[0], model.hidden_size), dtype=torch.float32).to(device)
            for i in range(X.shape[1]):
                with torch.set_grad_enabled(True):
                    hidden = model(X[:, i].unsqueeze(1), movie, hidden)
                    states[i] = hidden
            states = states.permute([1, 0, 2])
            tmp = torch.zeros((states.shape[0], states.shape[2]), dtype=torch.float32).to(device)
            for i, j in enumerate(last_word_indices):
                tmp[i] = states[i][j]
            pred = classfication_model(tmp)
            predict = torch.argmax(pred, dim=1)
            assert predict.shape == Y.shape
            predict = predict.cpu().numpy()
            Y = Y.cpu().numpy()
            for i in range(len(predict)):
                if Y[i] == 0:
                    correct_test += int(predict[i] == 0)
                elif Y[i] == 4:
                    correct_test += int(predict[i] == 4)
                else:
                    correct_test += int(predict[i] == 1 or predict[i] == 2 or predict[i] == 3)
            total_test += predict.shape[0]
        accuracy_test = float(correct_test) / float(total_test)
        test_acc.append(accuracy_test)
        print('epoch ', epoch, 'testing accuracy is ', accuracy_test)
        writer.add_scalar('att_acc_average_train', accuracy, epoch)
        writer.add_scalar('att_acc_average_test', accuracy_test, epoch)
        if accuracy == max(train_acc):
            torch.save(model.state_dict(), name + ' train_acc_max_att_class.pkl')
        if accuracy_test == max(test_acc):
            torch.save(model.state_dict(), name + ' test_acc_max_att_class.pkl')
        # 早停止
        case = 1
        if len(train_acc) > 10:
            for i in range(10):
                if train_acc[-1] >= train_acc[-2 - i]:
                    case = 0
        if case == 1 and len(train_acc) > 10:
            break


train_classficationAttnGRU(model_clf, classfication_model, optimizer1, optimizer2, corpus_indice, movies_indice, labels,
                           batch_size, length)

with open(name + 'att_early_class_training_process.txt', 'a') as f:
    f.write('\n' + 'classification训练过程参数变化')
    f.write('\n' + 'train_acc:{}'.format(train_acc))
    f.write('\n' + 'test_acc:{}'.format(test_acc))
    f.write('\n' + 'train_loss:{}'.format(class_loss_dict))


def get_accuracy(model, data, movie, last_word_indice, label):
    output = model(data, movie, last_word_indice)
    predict = torch.argmax(output, dim=1)
    label = label.squeeze()
    assert predict.shape == label.shape
    accuracy = torch.mean(label == predict)
    return accuracy


