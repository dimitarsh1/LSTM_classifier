# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.set_num_threads(8)
torch.manual_seed(1)

import codecs 

import NN.lstm_classifier as lstm_classifier

import argparse
import os

def prepare_sequence(seq, to_idx):
    idxs = [to_idx[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)
    
def load_vocabulary(vocfile):
    vocabulary_to_idx = {}
    with codecs.open(vocfile, 'r', 'utf8') as fh:
        for line in fh:
            for voc in line.split():
                if not voc in vocabulary_to_idx:
                    vocabulary_to_idx[voc] = len(vocabulary_to_idx)
    
    return vocabulary_to_idx
    
def main():
    parser = argparse.ArgumentParser(description='Clasifying a sentence with LSTM.')
    parser.add_argument('-m', '--model', required=True, help='the path to the model file.')
    parser.add_argument('-s', '--sentence', required=True, help='the sentence to classify')
    parser.add_argument('-v', '--vocabulary', required=True, help='the vocabulary.')

    args = parser.parse_args()
    
    if os.path.exists(args.vocabulary):
        vocabulary_to_idx = load_vocabulary(args.vocabulary)
    else:
        print("Path not found: ", args.vocabulary)
        exit(1)
    
    if os.path.exists(args.model):
        model = torch.load(args.model)
        model.eval()
        
        sentence_in = prepare_sequence(args.sentence.split(), vocabulary_to_idx)
        prob_logs = model(sentence_in)
        print(prob_logs)
        
    else:
        print("Path not found: ", args.model)
        exit(1)    

if __name__ == "__main__":
    main()
    
