import helper
import numpy as np
from data_analysis import get_avg_words_in_lines, get_unique_words, get_words_in_lines_range

def main():
    data_dir = './data/Seinfeld_Scripts.txt'
    text = helper.load_data(data_dir)

    #Visualize the text
    print('Dataset Stats')
    word_map = get_unique_words(text=text)
    print('Roughly the number of unique words: {}'.format(len(word_map)))
    get_avg_words_in_lines(text=text)
    line_range = (0,10)
    get_words_in_lines_range(text=text,line_range=line_range)

    #Pre-processing the data

if __name__=="__main__":
    main()