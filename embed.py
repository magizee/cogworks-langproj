import re, string
from typing import List
import itertools
from collections import Counter

punc_regex = re.compile('[{}]'.format(re.escape(string.punctuation)))

def strip_punc(corpus):
    """ Removes all punctuation from a string.
        
    Parameters
    ----------
    corpus : str

    Returns
    -------
    str
        the corpus with all punctuation removed"""
    # substitute all punctuation marks with ""
    return punc_regex.sub('', corpus)

def to_token(caption: str) -> List[str]:
    """ Makes captions/queries into tokens by word.

    Process captions/queries by lowercasing the text, removing punctuation, 
    and tokenizing words based on white space.

    Parameters
    ----------
    caption: str

    Returns
    -------
    list
        the caption is split into individual words"""
    return sorted(strip_punc(caption.lower()).split())

def make_the_vocab(token_list: List[List[str]]) -> List[str]:
    """ Primary code block that enables the making of a vocab list of all words in COCO dataset

    Parameters
    ----------
    ls: List[List[str]]
        Each inner list holds the tokens of one caption from the COCO dataset
        Outer list encompasses all the inner lists

    Returns
    -------
    list
        all the words are compiled into one list"""
    return sorted(set(itertools.chain(*token_list)))

def to_counter(doc):
    #splitting cospur into tokens
    token = sorted(to_token(doc))
    #counting tokens
    ls = Counter(token)
    #ls = Counter(dict(sorted(ls.items())))
    return ls

#Working on these functions: to_idf, to_glove, and caption_to_descriptor