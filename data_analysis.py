#####################################
# Name: Gopal Krishna, Gourav Beura #
# Course: CS 7180                   #
# Date: 11/8/2022                   #
#####################################

import numpy as np

def get_unique_words(text):
    if text==None:
        return "Data not available"

    word_map = {}

    for word in text.split():
        if word in word_map:
            word_map[word]+=1
        else:
            word_map[word]=1
    return word_map

def get_avg_words_in_lines(text):
    if text==None:
        return "Data not available"

    lines = text.split('\n')
    num_lines = len(lines)

    print(f'Number of Lines: {num_lines}')

    word_count_line = [len(line.split()) for line in lines]
    print('Average number of words in each line: {}'.format(np.average(word_count_line)))

def get_words_in_lines_range(text,line_range):
    print('The lines {} to {}:'.format(*line_range))
    print('\n'.join(text.split('\n')[line_range[0]:line_range[1]]))
