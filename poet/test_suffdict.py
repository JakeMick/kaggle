from __future__ import division
import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import poetry
import re
from nltk.corpus import cmudict
from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import CMUDictCorpusReader
import conf

conf.set_conf()

d = cmudict.dict()
suffdict = LazyCorpusLoader(
    'cmusuffdict', CMUDictCorpusReader, ['cmusuffdict'])
suffdict = suffdict.dict()


def suffdict_phonemes(word):
    # Use my cmu-based last syllable dictionary
    if re.search("((?i)[bcdfghjklmnpqrstvwxz]{1,2}[aeiouy]+[bcdfghjklmnpqrstvwxz]*(e|ed)?('[a-z]{1,2})?)(?![a-zA-Z]+)", word.lower()):
        last_syl = re.search("((?i)[bcdfghjklmnpqrstvwxz]{1,2}[aeiouy]+[bcdfghjklmnpqrstvwxz]*(e|ed)?('[a-z]{1,2})?)(?![a-zA-Z]+)", word.lower()).group()
        if last_syl in suffdict:
            return suffdict[last_syl][0]
        # else try without the first letter
        elif last_syl[1 - len(last_syl):] in suffdict:
            return suffdict[last_syl[1 - len(last_syl):]][0]
        # else try without the first 2 letters
        elif last_syl[2 - len(last_syl):] in suffdict:
            return suffdict[last_syl[2 - len(last_syl):]][0]
        # else try without the last 2 letters, if it ends in 's
        elif last_syl[-2:] == "'s":
            if last_syl[:-2] in suffdict:
                return suffdict[last_syl[:-2]][0].append('Z')
            elif last_syl[1 - len(last_syl):-2] in suffdict:
                return suffdict[last_syl[1 - len(last_syl):-2]][0].append('Z')
            elif last_syl[2 - len(last_syl):-2] in suffdict:
                return suffdict[last_syl[2 - len(last_syl):-2]][0].append('Z')
            else:
                return False
        # else try without the last letter, if it ends in s
        elif last_syl[-1] == "s":
            if last_syl[:-1] in suffdict:
                return suffdict[last_syl[:-1]][0].append('Z')
            elif last_syl[1 - len(last_syl):-1] in suffdict:
                return suffdict[last_syl[1 - len(last_syl):-1]][0].append('Z')
            elif last_syl[2 - len(last_syl):-1] in suffdict:
                return suffdict[last_syl[2 - len(last_syl):-1]][0].append('Z')
            else:
                return False
        else:  # If not in cmudict or my cmusuffdict
            return False
    else:
        return False


def cmu_phonemes(word):
    # If in cmudict, just use cmudict
    if not word.lower() in d:
        return False
    else:
        return min(d[word.lower()], key=len)


hit = 0
miss = 0

for word, vals in d.iteritems():
    cmu = cmu_phonemes(word)
    suff = suffdict_phonemes(word)
    if cmu and suff and poetry.rhyme_from_phonemes(cmu, suff):
        hit += 1
    elif not cmu:
        print "Not in cmudict!"
    elif not suff:
        print "Not in suffdict!"
    else:
        print word
        miss += 1

print "hits: "
print hit
print "misses: "
print miss
print "Percent accuracy: "
print (hit / (hit + miss)) * 100

# 90.8518971848
