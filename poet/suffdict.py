from __future__ import division
import re
from os import path
from curses.ascii import isdigit
import nltk
from nltk.corpus import cmudict
import conf

conf.set_conf()
cmu = cmudict.dict()
entries = cmudict.entries()

def suff(dict):
    f = open('suff_a.txt', 'a')
    for word, vals in dict.iteritems():
        if re.search("((?i)[BCDFGHJKLMNPQRSTVWXZ]{1,2}[AEIOUY]+[BCDFGHJKLMNPQRSTVWXZ]*(E|ED)?('[A-Z]{1,2})?)(?![a-zA-Z]+)", word):
            graphemes = re.search("((?i)[BCDFGHJKLMNPQRSTVWXZ]{1,2}[AEIOUY]+[BCDFGHJKLMNPQRSTVWXZ]*(E|ED)?('[A-Z]{1,2})?)(?![a-zA-Z]+)", word).group()
        val = min(vals, key=len)
        # for val in vals:
        i = -1
        while i >= 0 - len(val):
            if isdigit(val[i][-1]):
                str = " ".join(val[i:])
                f.write(graphemes + ' ' + str + '\n')
                f.write(graphemes[1 - len(graphemes):] + ' ' + str + '\n')
                break
            i -= 1
    f.close()

def most_prob(file):
    uniq_suffs = []
    goal = open('suff_c.txt', 'a')
    with open(file) as f:
        for line in f:
            suff = re.search("\s[a-zA-Z']+\s", line).group()
            if suff not in uniq_suffs:
                uniq_suffs.append(suff)
                new_line = re.sub("\d+\s(?=[a-z])", "", line)
                new_line = re.sub("(?<=[a-z])\s(?=[A-Z])", " 1 ", new_line).strip()
                goal.write(new_line + '\n')
    goal.close()


# First, I grabbed all last syllable graphones:
# suff(cmu)

# Then I sorted the results into unique types with frequency counts in the command line with:
#     sort < suff_a.txt | uniq -c | sort -nr > suff_b.txt

# Then I cut out all but the most likely of each possible phoneme set for each unique grapheme set,
# and put it into cmudict-like format:
# most_prob("suff_b.txt")
