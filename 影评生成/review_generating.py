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
name = 'tiaocan1'

rate_to_train = 1
training_data_name = 'train_set.txt'
testing_data_name = 'test_set.txt'
onehot_dimension = 300

is_beamsearch = False
embedding_dim = 300

# 调整
hidden_dim = 800
lr = 1e2
momentum = 0.1
batch_size = 32
#

num_epoch = 150
use_gpu = True
num_layers = 1
bidirectional = False

verbose = 1
#########################################################################


class beamNode():
    def __init__(self, hidden_state, word_indices, logp):
        self.hidden_state = hidden_state
        self.word_indices = word_indices
        self.logp = logp
    # def eval(self):
    #     return self.logp / (len(self.word_indices) + 1e-6)


# 加载电影名，并生成movies onehot
def transform_move_to_onehot(movies):
    movie_to_idx = {}
    idx_to_movie = {}
    movie_to_onehot = np.zeros((len(movie_names), onehot_dimension))
    for i, name in enumerate(movies):
        onehot = np.zeros(onehot_dimension)
        onehot[i] = 1
        movie_to_idx[name] = i
        movie_to_onehot[i] = onehot
        idx_to_movie[i] = name
    return movie_to_onehot, movie_to_idx, idx_to_movie


# 电影名读取
movies = pd.read_csv('movie_names_20.csv')
a = movies['movie_names']
movie_names = list(movies['movie_names'])
print(movie_names)
movie_to_onehot, movie_to_idx, idx_to_movie = transform_move_to_onehot(movie_names)

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
    corpus_test = list(map(lambda x: [word_to_idx[w] for w in x if w in word_to_idx], essays[num_train:]))
    topics_test = list(map(lambda x: [word_to_idx[w] for w in x if w in word_to_idx], topics[num_train:]))
    movies_test = list(map(lambda x: [movie_to_idx[x]], movies[num_train:]))
    labels = np.array(list(map(lambda x: [word_to_label[w] for w in x], topics[:num_train]))).squeeze()
    labels_test = np.array(list(map(lambda x: [word_to_label[w] for w in x], topics[num_train:]))).squeeze()
    return corpus_indice, topics_indice, movies_indice, corpus_test, topics_test, movies_test, labels, labels_test


corpus_indice, topics_indice, movies_indice, corpus_test, topics_test, movies_test, labels, labels_test = divide_data(
    essays, movies, topics, rate_to_train)

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


class TATLSTM(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_layers, movie_to_onehot, weight, num_labels, bidirectional,
                 dropout=0.5, **kwargs):
        super(TATLSTM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.bidirectional = bidirectional
        self.movie_to_onehot = movie_to_onehot
        if num_layers <= 1:
            self.dropout = 0
        else:
            self.dropout = dropout
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.rnn = nn.GRU(input_size=3 * self.embed_dim, hidden_size=self.hidden_dim,
                          num_layers=self.num_layers, bidirectional=self.bidirectional,
                          dropout=self.dropout).to(device)
        if self.bidirectional:
            self.decoder = nn.Linear(
                hidden_dim * 2, 1000).to(device)
        else:
            self.decoder = nn.Linear(
                hidden_dim, 1000).to(device)

    def forward(self, inputs, movie, topics, hidden=None):
        embeddings = self.embedding(inputs)
        topics_embed = self.embedding(topics)
        movie = self.movie_to_onehot[movie]
        movie = torch.tensor(movie.reshape((-1, 1, onehot_dimension)), dtype=torch.float32).to(device)
        topics_embed = torch.cat((movie, topics_embed), 2)
        topics_embed = topics_embed.expand(embeddings.shape[0], embeddings.shape[1], topics_embed.shape[2])
        embeddings = torch.cat((embeddings, topics_embed), 2)
        self.rnn.flatten_parameters()
        if hidden is None:
            states, hidden = self.rnn(embeddings.permute([1, 0, 2]))
        else:
            states, hidden = self.rnn(
                embeddings.permute([1, 0, 2]), hidden)
        outputs = self.decoder(states.reshape((-1, states.shape[-1])))
        # outputs = self.last(outputs)
        return (outputs, hidden)

    def init_hidden(self, num_layers, batch_size, hidden_dim, **kwargs):
        hidden = torch.zeros(num_layers, batch_size, hidden_dim)
        return hidden


def predict_rnn(movie, topics, num_chars, model, idx_to_word, word_to_idx):
    if is_beamsearch:
        return beamSearch(topk=3, model=model, movie=movie, topics=topics, max_length=num_chars, idx_to_word=idx_to_word,
                   word_to_idx=word_to_idx, movie_to_idx=movie_to_idx)
    else:
        output = [1]
        movie = [movie_to_idx[movie[0]]]
        movie = torch.tensor(movie).to(device)
        movie = movie.reshape((1, movie.shape[0]))
        topics = [word_to_idx[x] for x in topics]
        topics = torch.tensor(topics)
        topics = topics.reshape((1, topics.shape[0])).to(device)
        hidden = torch.zeros(num_layers, 1, hidden_dim).to(device)
        for t in range(num_chars):
            X = torch.tensor(output[-1]).reshape((1, 1))
            #         X = torch.tensor(output).reshape((1, len(output)))
            if use_gpu:
                X = X.to(device)
            pred, hidden = model(X, movie, topics, hidden)
            pred = adaptive_softmax.predict(pred)
            if pred[-1] == 2:
                break
            else:
                output.append(int(pred[-1]))
        #             output.append(int(pred.argmax(dim=1)[-1]))
        return (''.join([idx_to_word[i] for i in output[1:]]), output[1:])


def beamSearch(topk, model, movie, topics, max_length, idx_to_word, word_to_idx, movie_to_idx):
    movie = [movie_to_idx[movie[0]]]
    movie = torch.tensor(movie).to(device)
    movie = movie.reshape((1, movie.shape[0])).to(device)
    topics = [word_to_idx[x] for x in topics]
    topics = torch.tensor(topics)
    topics = topics.reshape((1, topics.shape[0])).to(device)
    hidden = torch.zeros(num_layers, 1, hidden_dim).to(device)

    pred, hidden = model(torch.ones((1, 1), dtype=torch.long).to(device), movie, topics, hidden)
    pred = adaptive_softmax.log_prob(pred).squeeze()
    value, indice = torch.topk(pred, topk)
    beamNodes = []
    final_beamNodes = []
    for i in range(topk):
        beamNodes.append(beamNode(hidden, [1, int(indice[i])], value[i]))
    for epoch in range(max_length - 1):
        preds = torch.empty((len(beamNodes), model.num_labels)).to(device)
        hiddens = []
        for j in range(len(beamNodes)):
            pred, hidden = model(torch.tensor(beamNodes[j].word_indices[-1], dtype=torch.long).view((1, 1)).to(device),
                                 movie, topics, beamNodes[j].hidden_state)
            hiddens.append(hidden)
            pred = adaptive_softmax.log_prob(pred).squeeze()
            preds[j] = (pred + beamNodes[j].logp) / (epoch + 2)
        preds = preds.reshape((-1))
        values, indices = torch.topk(preds, topk)
        tmp_beamNodes = []
        for v, i in zip(values, indices):
            num = int(i / model.num_labels)
            i = int(i) % model.num_labels
            if (i == 2):
                topk -= 1
                tmp_word_indices = beamNodes[num].word_indices.copy()
                final_beamNodes.append(beamNode(hiddens[num], tmp_word_indices, v * (epoch + 2)))
            else:
                tmp_word_indices = beamNodes[num].word_indices.copy()
                tmp_word_indices.append(i)
                tmp_beamNodes.append(beamNode(hiddens[num], tmp_word_indices, v * (epoch + 2)))
        beamNodes = tmp_beamNodes
        if (topk == 0):
            break
    final_beamNodes.extend(beamNodes)
    output = list(final_beamNodes[0].word_indices)
    return (''.join([idx_to_word[i] for i in output[1:]]), output[1:])


def evaluation(predicted_comments, original_comments):
    references = original_comments
    candidates = predicted_comments
    score = corpus_bleu(references, candidates)
    return score


vocab_size = len(vocab)
device = torch.device('cuda:0')
loss_function = nn.CrossEntropyLoss()
adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
    1000, len(vocab), cutoffs=[round(vocab_size / 20), 4 * round(vocab_size / 20)]).to(device)

model_TAT = TATLSTM(hidden_dim=hidden_dim, embed_dim=embedding_dim, num_layers=num_layers,
                    num_labels=len(vocab), movie_to_onehot=movie_to_onehot, weight=word_vec,
                    bidirectional=bidirectional)
model_TAT.to(device)
optimizer = optim.SGD(model_TAT.parameters(), lr=lr, momentum=momentum)


loss_dict = []
writer = SummaryWriter()
train_bleu = []
test_bleu = []


def train_TATLSTM(model_TAT, corpus_indice, movies_indice, topics_indice, batch_size, length):
    print('开始训练')
    for epoch in range(num_epoch):
        print('轮数：', epoch)
        # start = time.time()
        num, total_loss = 1, 0
        #     if epoch == 5000:
        #         optimizer.param_groups[0]['lr'] = lr * 0.1
        data = tav_data_iterator(
            corpus_indice, movies_indice, topics_indice, batch_size, max(length) + 1)
        #     hidden = model.module.init_hidden(num_layers, batch_size, hidden_dim)
        for X, Y, mask, movie, topics, _ in tqdm(data):
            num += 1
            if use_gpu:
                X = X.to(device)
                Y = Y.to(device)
                mask = mask.to(device)
                topics = topics.to(device)
            optimizer.zero_grad()
            output, hidden = model_TAT(X, movie, topics)
            hidden.detach_()
            l, _ = adaptive_softmax(output, Y.t().reshape((-1,)))
            loss = -l.reshape((-1, X.shape[0])).t() * mask
            loss = loss.sum(dim=1) / mask.sum(dim=1)
            loss = loss.mean()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model_TAT.parameters(), 1e-2)
            optimizer.step()
            total_loss += loss.item()
        loss_dict.append(total_loss)
        if total_loss == min(loss_dict):
            params = model_TAT.state_dict()
            torch.save(params, name + ' loss_min_modelTAT.pkl')
            torch.save(adaptive_softmax.state_dict(), name + ' loss_min_adaptive.pkl')
        writer.add_scalar('loss', total_loss, epoch)

        scores_train = []
        scores_test = []
        if ((epoch + 1) % verbose == 0) or (epoch == (num_epoch - 1)):
            comments = ['一星级', '二星级', '三星级', '四星级', '五星级']
            with open('training_process.txt', 'a') as f:
                f.write('\n' + '训练轮数：{}'.format(epoch))
                for movie in movie_names:
                    for comment in comments:
                        (tmp, b) = predict_rnn([movie], [comment], 100, model_TAT, idx_to_word, word_to_idx)

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
                        (tmp, b) = predict_rnn([a[0]], [a[1]], 100, model_TAT, idx_to_word, word_to_idx)
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
            writer.add_scalar('bleu_average_train', mean_score_train, epoch)
            writer.add_scalar('bleu_average_test', mean_score_test, epoch)
            train_bleu.append(mean_score_train)
            test_bleu.append(mean_score_test)
            if mean_score_train == max(train_bleu):
                torch.save(model_TAT.state_dict(), name + ' train_bleu_max_modelTAT.pkl')
                torch.save(adaptive_softmax.state_dict(), name + ' train_bleu_max_adaptive.pkl')
            if mean_score_test == max(test_bleu):
                torch.save(model_TAT.state_dict(), name + ' test_bleu_max_modelTAT.pkl')
                torch.save(adaptive_softmax.state_dict(), name + ' test_bleu_max_adaptive.pkl')
    return model_TAT


# 模型训练与保存
model_TAT = train_TATLSTM(model_TAT, corpus_indice, movies_indice, topics_indice, batch_size, length)
torch.save(adaptive_softmax.state_dict(), 'adaptive.pkl')
torch.save(model_TAT.state_dict(), 'modelTAT.pkl')


print('\n' + '训练后结果：')
comments = ['一星级', '二星级', '三星级', '四星级', '五星级']
for movie in movie_names:
    for comment in comments:
        (tmp, b) = predict_rnn([movie], [comment], 100, model_TAT, idx_to_word, word_to_idx)
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


with open('training_process.txt', 'a') as f:
    f.write('\n' + 'review训练过程参数变化')
    f.write('\n' + 'train_bleu:{}'.format(train_bleu))
    f.write('\n' + 'test_bleu:{}'.format(test_bleu))
    f.write('\n' + 'train_loss:{}'.format(loss_dict))
