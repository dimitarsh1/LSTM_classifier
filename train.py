# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import codecs

import argparse

import NN.lstm_classifier as lstm_classifier

import os
import sys
import random

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

use_gpu = torch.cuda.is_available()

def prepare_sequence(seq, to_idx):
    idxs = [to_idx[w] for w in seq]
    tensor = torch.cuda.LongTensor(idxs)
    return autograd.Variable(tensor)
    
def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def train(train_data, test_data, dev_data, vocabulary_to_idx, labels_to_idx):
    
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 50
    EPOCH = 10
    BATCH_SIZE = 10 
    
    best_dev_acc = 0.0

    model = lstm_classifier.LSTMClassifier(embedding_dim=EMBEDDING_DIM, 
                   hidden_dim=HIDDEN_DIM,
                   vocab_size=len(vocabulary_to_idx),
                   label_size=len(labels_to_idx),
                   batch_size=BATCH_SIZE,
                   device=)

    model.cuda() #EVA
    loss_function = nn.BCELoss() #change to BCEloss? binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    updates = 0
    for i in range(EPOCH):
        
        # Now let's shuffle the data to get some randomness
        random.seed(1)
        random.shuffle(train_data)
        
        print('epoch: %d start!' % i)
        train_epoch(model, 
                    train_data, 
                    loss_function, 
                    optimizer, 
                    vocabulary_to_idx, 
                    labels_to_idx, 
                    i)
        
        print('now best dev acc:',best_dev_acc)
        
        dev_acc = evaluate(model, dev_data, loss_function, vocabulary_to_idx, labels_to_idx, 'dev')
        test_acc = evaluate(model, test_data, loss_function, vocabulary_to_idx, labels_to_idx, 'test')
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            if os.listdir('models/'):
                os.system('rm models/best_model_minibatch_acc_*.model')
                
            print('New Best Dev!!!')
            torch.save(model, 'models/best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
            updates = 0
        else:
            updates += 1       # Early stopping criteria 
            if updates >= 10:
                break          # stop loop (finishes after 10 updates, or if epochs are finished (100)

def train_epoch(model, train_data, loss_function, optimizer, vocabulary_to_idx, label_to_idx, i):
    #enable training mode
    model.train()
    
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for batch in train_data:
        sent, label = batch #EVA
        truth_res.append(label_to_idx[label])
        model.batch_size = len(label.data) #EVA??
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = prepare_sequence(sent, vocabulary_to_idx)
        label = prepare_sequence([label], label_to_idx)
        
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        
        # nulify gradients not to be accumulated.
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        if count % 500 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count, loss.data[0]))

        loss.backward()
        optimizer.step()

    avg_loss /= len(train_data)
    print('epoch: %d done! \n train avg_loss:%g , acc:%g'%(i, avg_loss, get_accuracy(truth_res,pred_res)))


def evaluate(model, data, loss_function, vocabulary_to_idx, label_to_idx, name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for sent, label in data:
        truth_res.append(label_to_idx[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = prepare_sequence(sent, vocabulary_to_idx)
        label = prepare_sequence([label], label_to_idx)
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        pred_res.append(pred_label)
        # model.zero_grad() # should I keep this when I am evaluating the model?
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
    return acc


def load_data(datafile):
    data = []
    with codecs.open(datafile, 'r', 'utf8') as fh:
        for line in fh:
            data += [(line.split()[:-1], line.split()[-1])]
    
    return data
    
def load_vocabulary(vocfile):
    vocabulary_to_idx = {}
    with codecs.open(vocfile, 'r', 'utf8') as fh:
        for line in fh:
            for voc in line.split():
                if not voc in vocabulary_to_idx:
                    vocabulary_to_idx[voc] = len(vocabulary_to_idx)
    
    return vocabulary_to_idx
    
def main():
    ''' read arguments from the command line and initiate the training.
    '''

    parser = argparse.ArgumentParser(description='Train a basic LSTM sentence classifier.')
    parser.add_argument('-v', '--vocabulary', required=True, help='the vocabulary.')
    parser.add_argument('-l', '--labels', required=True, help='the labels vocabulary.')
    parser.add_argument('-t', '--train', required=True, help='the train set.')
    parser.add_argument('-e', '--test', required=True, help='the test set.')
    parser.add_argument('-d', '--dev', required=True, help='the dev set.')
    
    args = parser.parse_args()
    
    # Read data first:
    # Todo: make this more efficient
    train_data = []
    test_data = []
    dev_data = []
    
    vocabulary_to_idx = {}
    labels_to_idx = {}
    
    if os.path.exists(args.train):
        train_data = load_data(args.train)   
    else:
        print("ERROR: Path not found: ", args.train, file=sys.stderr)
        exit(1)    
    
    if os.path.exists(args.test):
        test_data = load_data(args.test)    
    else:
        print("ERROR: Path not found: ", args.test, file=sys.stderr)
        exit(1)
        
    if os.path.exists(args.dev):
        dev_data = load_data(args.dev)
    else:
        print("ERROR: Path not found: ", args.dev, file=sys.stderr)
        exit(1)   
    
    if os.path.exists(args.vocabulary):
        vocabulary_to_idx = load_vocabulary(args.vocabulary)
    else:
        print("ERROR: Path not found: ", args.vocabulary, file=sys.stderr)
        exit(1)  
    
    if os.path.exists(args.labels):
        labels_to_idx = load_vocabulary(args.labels)
    else:
        print("ERROR: Path not found: ", args.labels, file=sys.stderr)
        exit(1)  
    
    train(train_data, test_data, dev_data, vocabulary_to_idx, labels_to_idx)


if __name__ == "__main__":
    main()
