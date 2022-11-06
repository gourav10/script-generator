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