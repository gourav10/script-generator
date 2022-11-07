import numpy as np
import torch
import torch.nn as nn
from rnn import RNN
from torch.utils.data import TensorDataset, DataLoader


class Engine:
    def __init__(self,
                 vocab_size, 
                 output_size, 
                 embedding_dim, 
                 hidden_dim, 
                 n_layers, 
                 dropout=0.5,
                 learning_rate = 0.0009,) -> None:
        self.model = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout)
        if(torch.cuda.is_available()):
            self.model.to(device='cuda')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        pass

    def batch_data(self, words, sequence_length, batch_size):
        """
        Batch the neural network data using DataLoader
        :param words: The word ids of the TV scripts
        :param sequence_length: The sequence length of each batch
        :param batch_size: The size of each batch; the number of sequences in a batch
        :return: DataLoader with batched data
        """
        # TODO: Implement function
        words = np.array(words)

        features, targets = [], []
        
        try:
            for i in range(len(words)-sequence_length):
                features.append(words[i:i+sequence_length])
                targets.append(words[i+sequence_length])
        except IndexError:
            pass
        feature_tensors, target_tensors = np.array(features), np.array(targets)
        
        feature_tensors, target_tensors = torch.from_numpy(feature_tensors), torch.from_numpy(target_tensors)
        
        data = TensorDataset(feature_tensors, target_tensors)
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        
        return data_loader
        
    def forward_back_prop(self,optimizer,criterion,inp,target,hidden):
        """
        Forward and backward propagation on the neural network
        :param rnn: The PyTorch Module that holds the neural network
        :param optimizer: The PyTorch optimizer for the neural network
        :param criterion: The PyTorch loss function
        :param inp: A batch of input to the neural network
        :param target: The target output for the batch of input
        :return: The loss and the latest hidden state Tensor
        """
        if (torch.cuda.is_available()):
            inp, target = inp.to('cuda'), target.to('cuda')

        hidden = tuple([each.data for each in hidden])

        self.model.zero_grad()
        output, h = self.model(inp, hidden)
        loss = criterion(output, target.long())
        loss.backward()
        optimizer.step()
        return loss.item(), h

    def train_model(self, train_dl, batch_size, n_epochs, show_every_n_batches=100):
        batch_losses = []
    
        self.model.train()

        print("Training for %d epoch(s)..." % n_epochs)
        for epoch_i in range(1, n_epochs + 1):
            
            # initialize hidden state
            hidden = self.model.init_hidden(batch_size)
            
            for batch_i, (inputs, labels) in enumerate(train_dl, 1):
                
                # make sure you iterate over completely full batches, only
                n_batches = len(train_dl.dataset)//batch_size
                if(batch_i > n_batches):
                    break
                
                # forward, back prop
                loss, hidden = self.forward_back_prop(self.optimizer,self.criterion, inputs, labels, hidden)          
                # record loss
                batch_losses.append(loss)

                # printing loss stats
                if batch_i % show_every_n_batches == 0:
                    print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                        epoch_i, n_epochs, np.average(batch_losses)))
                    batch_losses = []

        # returns a trained rnn
        return self.model
