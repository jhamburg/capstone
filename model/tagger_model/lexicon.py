import pandas as pd
import numpy as np
import os
import pickle


class lexiconTransformer():
    """Create a lexicon and transform sentences and tags
       to indexes for use in the model."""
    
    def __init__(self, words_min_freq = 1, tags_min_freq = 1, 
                 savePath = 'models', saveNamePrefix='', 
                 unknown_word_token = u'<UNK>',
                 unknown_tag_token = u'<UNK>'):
        self.words_min_freq = words_min_freq
        self.tags_min_freq = tags_min_freq
        self.words_lexicon = None
        self.unknown_word_token = unknown_word_token
        self.indx_to_words_dict = None
        self.tags_lexicon = None
        self.unknown_tag_token = unknown_tag_token
        self.indx_to_tags_dict = None
        self.savePath = savePath
        self.saveNamePrefix = saveNamePrefix
    
    def fit(self, sents, tags):
        """Create lexicon based on sentences and tags"""
        self.make_words_lexicon(sents)
        self.make_tags_lexicon(tags)
        
        self.make_lexicon_reverse()
        self.save_lexicon()
                
    def transform(self, sents, tags):
        sents_indxs = self.tokens_to_idxs(sents, self.words_lexicon)
        tags_indxs = self.tokens_to_idxs(tags, self.tags_lexicon)
        return (sents_indxs, tags_indxs)

    def fit_transform(self, sents, tags):
        self.fit(sents, tags)
        return self.transform(sents, tags)
        
    def make_words_lexicon(self, sents_token):
        """Wrapper for words lexicon"""
        self.words_lexicon = self.make_lexicon(sents_token, self.words_min_freq,
                                               self.unknown_word_token)

    def make_tags_lexicon(self, tags_token):
        """Wrapper for tags lexicon"""
        self.tags_lexicon = self.make_lexicon(tags_token, self.tags_min_freq,
                                              unknown = self.unknown_tag_token)

    def make_lexicon(self, token_seqs, min_freq=1, unknown = u'<UNK>'):
        """Create lexicon from input based on a frequency

            Parameters:
            
            token_seqs
            ----------
               A list of a list of input tokens that will be used to create the lexicon
            
            min_freq
            --------
               Number of times the token needs to be in the corpus to be included in the
               lexicon.  Otherwise, will be replaced with the "unknown" entry
            
            unknown
            -------
               The word in the lexicon that should be used for tokens not existing in lexicon.
               This can be a value that already exists in input list.  For instance, in 
               Named Entity Recognition, a value of "other" or "O" may already be a tag 
               and so having "other" and "unknown" are the same thing!
        """
        # Count how often each word appears in the text.
        token_counts = {}
        for seq in token_seqs:
            for token in seq:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1

        # Then, assign each word to a numerical index. 
        # Filter words that occur less than min_freq times.
        lexicon = [token for token, count in token_counts.items() if count >= min_freq]
        
        # Have to delete unknown value from token list so not a gap in lexicon values when
        # turning it into a lexicon (aka, if unknown == OTHER and that is the 7th value, 
        # then 7 won't exist in the lexicon which may cause issues)
        if unknown in lexicon:
            lexicon.remove(unknown)

        # Indices start at 1. 0 is reserved for padding, and 1 is reserved for unknown words.
        lexicon = {token:idx + 2 for idx,token in enumerate(lexicon)}
        
        lexicon[unknown] = 1 # Unknown words are those that occur fewer than min_freq times
        lexicon_size = len(lexicon)
        return lexicon
    
    def save_lexicon(self):
        "Save lexicons by pickling them"
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        with open(os.path.join(self.savePath, self.saveNamePrefix + 'words_lexicon.pkl'), 'wb') as f:
            pickle.dump(self.words_lexicon, f)
            
        with open(os.path.join(self.savePath, self.saveNamePrefix + 'tags_lexicon.pkl'), 'wb') as f:
            pickle.dump(self.tags_lexicon, f)
            
    def load_lexicon(self):
        with open(os.path.join(self.savePath, self.saveNamePrefix + 'words_lexicon.pkl'), 'rb') as f:
            self.words_lexicon = pickle.load(f)
            
        with open(os.path.join(self.savePath, self.saveNamePrefix + 'tags_lexicon.pkl'), 'rb') as f:
            self.tags_lexicon = pickle.load(f)
        
        self.make_lexicon_reverse()
        
    def make_lexicon_reverse(self):
        self.indx_to_words_dict = self.get_lexicon_lookup(self.words_lexicon)
        self.indx_to_tags_dict = self.get_lexicon_lookup(self.tags_lexicon)
    
    def get_lexicon_lookup(self, lexicon):
        '''Make a dictionary where the string representation of 
           a lexicon item can be retrieved from its numerical index'''
        lexicon_lookup = {idx: lexicon_item for lexicon_item, idx in lexicon.items()}
        return lexicon_lookup
    
    def tokens_to_idxs(self, token_seqs, lexicon):
        """Transform tokens to numeric indexes or <UNK> if doesn't exist"""
        idx_seqs = [[lexicon[token] if token in lexicon else lexicon['<UNK>'] for 
                                 token in token_seq] for token_seq in token_seqs]
        return idx_seqs