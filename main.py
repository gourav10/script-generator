import helper
import numpy as np
import data_analysis as da

from engine import Engine

def main():
    data_dir = './data/Seinfeld_Scripts.txt'
    text = helper.load_data(data_dir)

    #Visualize the text
    print('*****Dataset Stats*****')
    word_map = da.get_unique_words(text=text)
    print('Roughly the number of unique words: {}'.format(len(word_map)))
    da.get_avg_words_in_lines(text=text)
    line_range = (0,10)
    da.get_words_in_lines_range(text=text,line_range=line_range)

    #Pre-processing the data
    helper.preprocess_and_save_data(text,helper.token_lookup,helper.create_lookup_tables)

    #Load from checkpoint
    int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
    print('***********************')

    #Define Hyper-parameters
    # Sequence Length
    sequence_length = 20 # of words in a sequence
    # Batch Size
    batch_size = 256

    # Training parameters
    # Number of Epochs
    num_epochs = 20
    # Learning Rate
    learning_rate = 0.0009

    # Model parameters
    # Vocab size
    vocab_size = len(vocab_to_int)
    # Output size
    output_size = len(vocab_to_int)
    # Embedding Dimension
    embedding_dim = 500
    # Hidden Dimension
    hidden_dim = 256
    # Number of RNN Layers
    n_layers = 3

    # Show stats for every n number of batches
    show_every_n_batches = 500

    engine = Engine(vocab_size,output_size,embedding_dim,hidden_dim,n_layers)
    
    #Test Dataloader
    test_text = range(50)
    test_dl = engine.batch_data(test_text,sequence_length=5, batch_size=10)
    data_iter = iter(test_dl)
    sample_x, sample_y = next(data_iter)
    print("***** Test Dataloader *****")
    print(sample_x.shape)
    print(sample_x)
    print()
    print(sample_y.shape)
    print(sample_y)
    print("***************************")

    train_loader = engine.batch_data(int_text, sequence_length, batch_size)


    trained_rnn = engine.train_model(train_loader, batch_size, num_epochs, show_every_n_batches)
    helper.save_model('./save/trained_rnn', trained_rnn)
    print('Model Trained and Saved')
    
if __name__=="__main__":
    main()