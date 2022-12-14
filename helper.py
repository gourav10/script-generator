#####################################
# Name: Gopal Krishna, Gourav Beura #
# Course: CS 7180                   #
# Date: 11/8/2022                   #
#####################################

import os
import pickle
import torch
from collections import Counter

SPECIAL_WORDS = {'PADDING': '<PAD>'}

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    vocab_to_int, int_to_vocab = dict(), dict()
    counts = Counter(text)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}
    int_to_vocab = {ii: word for word, ii in vocab_to_int.items()}

    # return tuple
    return (vocab_to_int, int_to_vocab)

def load_data(path):
    """
    Load Dataset from File
    """
    input_file = os.path.join(path)
    with open(input_file, "r") as f:
        data = f.read()

    return data

def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punc_dict = {'.': '||period||',
                 ',': '||comma||',
                 '"': '||quotes||',
                 ';': '||semicolon||',
                 '!': '||exclamation_mark||',
                 '?': '||question_mark||',
                 '(': '||left_parantheses||',
                 ')': '||right_parantheses||',
                 '-': '||dash||',
                 '\n': '||return||'}
    return punc_dict

def preprocess_and_save_data(text, token_lookup, create_lookup_tables):
    """
    Preprocess Text Data
    """
    # text = load_data(dataset_path)
    
    # Ignore notice, since we don't use it for analysing the data
    #text = text[81:]

    token_dict = token_lookup()
    for key, token in token_dict.items():
        text = text.replace(key, ' {} '.format(token))

    text = text.lower()
    text = text.split()

    vocab_to_int, int_to_vocab = create_lookup_tables(text + list(SPECIAL_WORDS.values()))
    int_text = [vocab_to_int[word] for word in text]
    pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))

def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    return pickle.load(open('preprocess.p', mode='rb'))

def save_model(filename, decoder):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)

def load_model(filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    return torch.load(save_filename)
