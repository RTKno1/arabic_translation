#James Chartouni
''' overrides the torchtext.data.field class so that Spacy can use a custom Tokenizer '''

#https://pytorch.org/text/_modules/torchtext/data/utils.html#get_tokenizer
#https://pytorch.org/text/_modules/torchtext/data/field.html 

import random
from contextlib import contextmanager
from copy import deepcopy
import re

from functools import partial
import spacy
from spacy.vocab import Vocab
from spacy.language import Language

from torchtext.data import Field

from collections import Counter, OrderedDict
from itertools import chain
import six
import torch
from tqdm import tqdm


class Custom_Field(Field):
    
    def __init__(self, sequential=True, use_vocab=True, init_token=None,
                 eos_token=None, fix_length=None, dtype=torch.long,
                 preprocessing=None, postprocessing=None, lower=False,
                 tokenize=None, tokenizer_language='en', include_lengths=False,
                 batch_first=False, pad_token="<pad>", unk_token="<unk>",
                 pad_first=False, truncate_first=False, stop_words=None,
                 is_target=False):
        self.sequential = sequential
        self.use_vocab = use_vocab
        self.init_token = init_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.fix_length = fix_length
        self.dtype = dtype
        self.preprocessing = preprocessing
        self.postprocessing = postprocessing
        self.lower = lower
        # store params to construct tokenizer for serialization
        # in case the tokenizer isn't picklable (e.g. spacy)
        self.tokenizer_args = (tokenize) # removed tokenizer_language 
        self.tokenize = custom_get_tokenizer(tokenize) #*** This is what I changed ****
        self.include_lengths = include_lengths
        self.batch_first = batch_first
        self.pad_token = pad_token if self.sequential else None
        self.pad_first = pad_first
        self.truncate_first = truncate_first
        try:
            self.stop_words = set(stop_words) if stop_words is not None else None
        except TypeError:
            raise ValueError("Stop words must be convertible to a set")
        self.is_target = is_target

    
    
  
    
    
    

def _spacy_tokenize(x, spacy):
    return [tok.text for tok in spacy.tokenizer(x)]  
    
    
def custom_get_tokenizer(tokenizer):
    r"""
    Generate tokenizer function for a string sentence.

    Arguments:
        tokenizer: the name of tokenizer function. If None, it returns split()
            function, which splits the string sentence by space.
            If basic_english, it returns _basic_english_normalize() function,
            which normalize the string first and split by space. If a callable
            function, it will return the function. If a tokenizer library
            (e.g. spacy, moses, toktok, revtok, subword), it returns the
            corresponding library.
        language: Default en

    Examples:
        >>> import torchtext
        >>> from torchtext.data import get_tokenizer
        >>> tokenizer = get_tokenizer("basic_english")
        >>> tokens = tokenizer("You can now install TorchText using pip!")
        >>> tokens
        >>> ['you', 'can', 'now', 'install', 'torchtext', 'using', 'pip', '!']

    """

  

    if tokenizer == "spacy":
        try:
   
            spm_vocab_data = open("data/model/spm.vocab", "r")

            spm_vocab_list = []

            for line in spm_vocab_data.readlines():
                spm_vocab_list.append(line.split("\t")[0])

            
            #spm_vocab = Vocab(strings=spm_vocab_list)
            #spacy_spm_tokenizer = Tokenizer(spm_vocab)
            
            spm_vocab = Vocab(strings=spm_vocab_list)
            nlp = Language(spm_vocab)
            print(nlp)
            
            spacy_lang = spacy.load(nlp)
            return partial(_spacy_tokenize, spacy=spacy_lang)
        except ImportError:
            print("Please install SpaCy. "
                  "See the docs at https://spacy.io for more information.")
            raise
        except AttributeError:
            print("Please install SpaCy and the SpaCy {} tokenizer. "
                  "See the docs at https://spacy.io for more "
                  "information.".format(language))
            raise
   