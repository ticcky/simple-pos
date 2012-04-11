#!/usr/bin/env python
#
# Simple Bigram POS Tagger
# author: Lukas Zilka
#

from optparse import OptionParser
import pickle
import json
import sys
import hmm
import re
from collections import defaultdict

def get_p_of(what, where):
    """Return probability of item `what` in the list of tuples `where`. 
    The items are of the form (probability, item_label)."""
    
    for freq, item in where:
        if item == what:
            return freq
    return 0.0

def make_stats_for(from_dict, to_dict, according_to):
    """Compute conditional probability table (CPT) out of the
    dictionary `from_dict`. `from_dict` has keys in form of tuples,
    and their values are frequencies in which they occurred.

    Arguments: 
     - `to_dict` - destination for the CPT 
     - `according_to` - rank of the variable which is conditioned (e.g. if we compute CPT
    for P(A|B), and the `from_dict` keys are in form of (a, b), then `according_to` has 
    value 1, because A is of rank 0, and B is of rank 1). """

    # aggregate
    for itms, freq in from_dict.items():
        to_dict[itms[according_to]] += [tuple([float(freq)] + [itm for i, itm in enumerate(itms) if i != according_to])]

    # normalize and sort
    for itm, itm_lst in to_dict.items():
        norm_factor = sum([freq for freq, i in itm_lst])
        to_dict[itm] = [(float(freq)/norm_factor, tag) for freq, tag in itm_lst] # normalize counts
        to_dict[itm] = sorted(to_dict[itm])[::-1] # sort so that highest probability is first


def pick_largest(dictnry):
    """Sorts items in the dictionary `dictnry` according to their value, and return the
    key of the largest one."""
    
    if len(dictnry) == 0:
        return None
    else:
        return sorted(map(tuple, map(reversed, dictnry.items())))[-1][1]

    
class TiccPOSTagger:
    NUMBER_PLACEHOLDER = "__number__"
    OOV_PLACEHOLDER = "oov"
    _serialized_fields = ['words', 'tags', 'word_tags', 'word_ptags', 'tag_tags', 'tag_priors']
    
    def __init__(self):
        self.words = []
        self.tags = []

        self.word_tag_freqs = defaultdict(lambda: 0) # freq(word,tag)
        self.tag_tag_freqs = defaultdict(lambda: 0) # freq(tag, prev_tag)
        self.word_ptag_freqs = defaultdict(lambda: 0) # freq(word, prev_tag)
        self.tag_freqs = defaultdict(lambda: 0) # freq(tag)

        self.number_re = re.compile("[0-9]")

        self.build_aux()

    def build_stats(self):
        """Compute conditional probabilities that we will use for determining the right 
        POS tags."""
        
        self.word_tags = defaultdict(list)
        self.word_ptags = defaultdict(list)
        self.tag_tags = defaultdict(list)
        make_stats_for(from_dict = self.word_tag_freqs, \
                       to_dict = self.word_tags, \
                       according_to = 0)

        make_stats_for(from_dict = self.word_ptag_freqs, \
                       to_dict = self.word_ptags, \
                       according_to = 0)
                       
        make_stats_for(from_dict = self.tag_tag_freqs, \
                       to_dict = self.tag_tags, \
                       according_to = 1)    

        self.compute_tag_priors()
        
    def compute_tag_priors(self):
        self.tag_priors = {}
        norm_value = sum(self.tag_freqs.values())
        
        for tag, freq in self.tag_freqs.items():
            self.tag_priors[tag] = float(self.tag_freqs[tag]) / norm_value

    def prepare_serialization(self):
        """Everything that needs to be done before the model is serialized and saved to
        file, should go here."""
        
        pass

    def build_aux(self):
        """Rebuilds some auxiliary structures that it is nice to have during the computations.
        Called after creation of the model and unserialization."""
        
        self.tag_to_ndx = dict(map(reversed, enumerate(self.tags))) # assign ids to tags
        self.word_to_ndx = dict(map(reversed, enumerate(self.words))) # assign ids to words

    def load_from_file(self, filename):
        f = open(filename, 'rb')
        data = pickle.loads(f.read())
        f.close()

        self.unserialize(data)
        self.build_aux()

    def save_to_file(self, filename):
        data = pickle.dumps(self.serialize())
        
        f = open(filename, 'wb')
        f.write(data)
        f.close()

    def serialize(self):
        """Serializes the model into a string. Can be saved to disk."""
        
        self.prepare_serialization()
        
        res = {}
        for field in self._serialized_fields:
            res[field] = getattr(self, field)
            
        return res

    def unserialize(self, data):
        """Loads the model attributes from a dictionary."""
        for field, value in data.items():
            setattr(self, field, value)

    def train(self, filename):
        """Estimate the model parameters from the data in `filename`."""
        
        last_word_id = last_tag_id = -1
        
        f = open(filename, 'rb')
        # in a loop feed the training procedure with the current and past training data
        for ln in f:
            word, tag = ln.strip().split('\t')
            word = self.preprocess_word(word)
            word_id, tag_id = self.get_word_id(word), self.get_tag_id(tag)
            
            self.process_training_sample(word_id, tag_id, last_word_id, last_tag_id)
            
            last_word_id, last_tag_id = word_id, tag_id

        f.close()

    def preprocess_word(self, word):
        """Transforms the word before it is used for training/tagging."""

        # if there is any number in the word, treat it as a number
        if self.number_re.search(word) != None:
            return self.NUMBER_PLACEHOLDER
        else:
            return word

    def process_training_sample(self, word_id, tag_id, last_word_id, last_tag_id):
        """Process one training sample, by updating the computed statistics. These
        are later used for building the conditional probability tables that are then
        utilized for tagging."""
        
        self.word_tag_freqs[(word_id, tag_id)] += 1
        self.word_ptag_freqs[(word_id, last_tag_id)] += 1
        self.tag_tag_freqs[(tag_id, last_tag_id)] += 1        
        self.tag_freqs[tag_id] += 1

    def get_word_id(self, word, add_if_not_present = True):
        return self.get_id_generic(word, self.word_to_ndx, self.words, add_if_not_present)

    def get_tag_id(self, word, add_if_not_present = True):
        return self.get_id_generic(word, self.tag_to_ndx, self.tags, add_if_not_present)

    def get_id_generic(self, value, mapping_dict, value_list, add_if_not_present):
        """Return `value`'s index in `mapping_dict` if it is there. If not, and 
        `add_if_not_present` is True, the `value` is assigned a new index and added to
        `value_list` and `mapping_dict`. Makes possible addressing words and tags by
        their indexes, instead of strings."""
        
        id = mapping_dict.get(value, None)

        # to get the id, we need to insert it into our structures first
        if id is None and add_if_not_present:
            mapping_dict[value] = len(value_list) # save id
            value_list.append(value) # save value            
            id = mapping_dict.get(value, None)
            
        return id

    def get_tag(self, word, last_word, last_tag):        
        """Try to tag `word`, given `last_word` and `last_tag` by the correct POS tag."""
        
        ret_word = word

        # try to lowercase the word if it has not been found in its original form
        word = self.preprocess_word(word)
        word_id = self.get_word_id(word, add_if_not_present = False)
        if word_id is None:
            word_id = self.get_word_id(word.lower(), add_if_not_present = False)
        
        last_word_id = self.get_word_id(word, add_if_not_present = False)
        last_tag_id = self.get_tag_id(last_tag, add_if_not_present = False)

        # if records for this word exist in our tables, try to tag it according to them,
        # otherwise we turn back to P(tag|prev_tag) and P(tag)
        if word_id is not None:
            # compute the probability as:
            #  P(tag|word,prev_tag) = P(tag|word)*P(tag|prev_tag)/P(word,prev_tag)
            possible_tags = {}        
            for tag in self.tags:
                tag_id = self.get_tag_id(tag)

                P_t_given_w = get_p_of(tag_id, self.word_tags[word_id])
                P_t_given_tp = get_p_of(tag_id, self.tag_tags[last_tag_id])
                P_w_and_tp= get_p_of(last_tag_id, self.word_ptags[word_id])

                if P_w_and_tp > 0:
                    possible_tags[tag_id] = P_t_given_w * P_t_given_tp / P_w_and_tp
                else:
                    possible_tags[tag_id] = P_t_given_w

            tag_id = pick_largest(possible_tags)

        else:
            # we say that we haven't seen the word and return tag with the highest 
            # P(tag|prev_tag) if it is known, otherwise just tag with highest P(tag)
            ret_word = self.OOV_PLACEHOLDER
            plist = self.tag_tags[last_tag_id]
            if len(plist) > 0:
                tag_id = plist[0][1]
            else:
                tag_id = pick_largest(self.tag_priors)
        
        return ret_word, self.tags[tag_id]
        

def train_tagger(tagger_file, train_file, debug = False):
    t = TiccPOSTagger()
    t.train(train_file)
    t.build_stats() # transform counted statistics to conditional probability tables
    t.save_to_file(tagger_file)

def run_tagger(tagger_file, input_file, debug = False):
    tagger = TiccPOSTagger()
    tagger.load_from_file(tagger_file)

    last_word = None
    last_tag = None

    f_in = open(input_file, 'r')
    for ln in f_in:
        # for each example, get its tag
        word, pos = ln.strip().split('\t')
        myword, mytag = tagger.get_tag(word, last_word, last_tag)
        
        last_word = word
        last_tag = mytag

        if not debug:
            print "%s\t%s" % (myword, mytag,)
        else:
            print "%s\t%s\t%s\t%d" % (myword, mytag, pos, pos == mytag)
    

if __name__ == "__main__":
    opt_parse = OptionParser()
    opt_parse.add_option("-t", "--train", dest = "train", default = False, 
                         action = "store_true",  help = "train the model")
    opt_parse.add_option("-d", "--data", dest = "data_file", help = "training data")
    opt_parse.add_option("-m", "--model", dest = "model_file", help = "file of the model")
    opt_parse.add_option("-x", "--debug", dest = "debug", default = False, action = "store_true", help = "debug mode")

    (opts, args) = opt_parse.parse_args()

    if opts.data_file is None or opts.model_file is None:
        opt_parse.error("not enough parameters")
    
    if opts.train:
        train_tagger(opts.model_file, opts.data_file, opts.debug)
    else:
        run_tagger(opts.model_file, opts.data_file, opts.debug)
        


