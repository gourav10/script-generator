import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn import RNN
from torch.utils.data import TensorDataset, DataLoader
import helper

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
        # if(torch.cuda.is_available()):
        #     self.model.to(device='cuda')

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
        # if (torch.cuda.is_available()):
        #     inp, target = inp.to('cuda'), target.to('cuda')

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

        min_loss = float('inf')

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
                    
                    if(min_loss>np.average(batch_losses)):
                        min_loss = np.average(batch_losses)
                        helper.save_model(f'./save/rnn_train', self.model)
                    batch_losses = []


        # returns a trained rnn
        return self.model

    def generate(self, sequence_length, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
        """
        Generate text using the neural network
        :param decoder: The PyTorch Module that holds the trained neural network
        :param prime_id: The word id to start the first prediction
        :param int_to_vocab: Dict of word id keys to word values
        :param token_dict: Dict of puncuation tokens keys to puncuation values
        :param pad_value: The value used to pad a sequence
        :param predict_len: The length of text to generate
        :return: The generated text
        """
        self.model.eval()
        
        # create a sequence (batch_size=1) with the prime_id
        current_seq = np.full((1, sequence_length), pad_value)
        current_seq[-1][-1] = prime_id
        predicted = [int_to_vocab[prime_id]]
        
        for _ in range(predict_len):
            if self.train_on_gpu:
                current_seq = torch.LongTensor(current_seq)
            else:
                current_seq = torch.LongTensor(current_seq)
            
            # initialize the hidden state
            hidden = self.model.init_hidden(current_seq.size(0))
            
            # get the output of the rnn
            output, _ = self.model(current_seq, hidden)
            
            # get the next word probabilities
            p = F.softmax(output, dim=1).data
            if(self.train_on_gpu):
                p = p.cpu() # move to cpu
            
            # use top_k sampling to get the index of the next word
            top_k = 5
            p, top_i = p.topk(top_k)
            top_i = top_i.numpy().squeeze()
            
            # select the likely next word index with some element of randomness
            p = p.numpy().squeeze()
            word_i = np.random.choice(top_i, p=p/p.sum())
            
            # retrieve that word from the dictionary
            word = int_to_vocab[word_i]
            predicted.append(word)     
            
            # the generated word becomes the next "current sequence" and the cycle can continue
            current_seq = np.roll(current_seq.cpu(), -1, 1)
            current_seq[-1][-1] = word_i
        
        gen_sentences = ' '.join(predicted)
        
        # Replace punctuation tokens
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
        gen_sentences = gen_sentences.replace('\n ', '\n')
        gen_sentences = gen_sentences.replace('( ', '(')
        
        # return all the sentences
        return gen_sentences