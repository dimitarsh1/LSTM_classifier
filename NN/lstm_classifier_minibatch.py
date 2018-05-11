import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class biLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dict_size, label_size, batch_size, deviceid=-1):
        super(biLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.deviceid = deviceid
                
        self.word_embeddings = nn.Embedding(dict_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim, label_size) 
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        if self.deviceid == -1:
            h0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim // 2))
            c0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim // 2))
        else:
            h0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim // 2).cuda())
            c0 = autograd.Variable(torch.zeros(1 * 2, self.batch_size, self.hidden_dim // 2).cuda())
        return (h0, c0)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size , -1)
        
        if self.deviceid > -1:
            x = x.cuda()
        
        self.lstm.flatten_parameters()
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs
