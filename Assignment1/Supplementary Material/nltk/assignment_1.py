# CS 565: Intelligent Systems and Interfaces
# Assignment 1
# Basic Text Processing

# imports
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.collocations import *
from nltk.tag import pos_tag

import operator
import sys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig


'''
IO Utility functions
'''
def printSep(n=50):
    str=""
    for i in range(n):
        str += "-"
    print str


def printSep2(n=50):
    str=""
    for i in range(n):
        str += "#"
    print str

def prettyprintugs(x):
    print "Unigram", "\t", "Frequency"
    for j in x:
       print j[0], "\t\t", j[1]

def prettyprintbgs(x):
    print "Bigram", "\t", "Frequency"
    for j in x:
       print j[0][0], j[0][1], "\t\t", j[1]

def prettyprintbgs_col(x):
    print "Bigram", "\t", "Collocations"
    for j in x:
       print j[0], "\t\t", j[1]

def prettyprinttgs(x):
    print "Trigram", "\t", "Frequency"
    for j in x:
       print j[0][0], j[0][1], j[0][2], "\t\t", j[1]

def prettyprinttgs_col(x):
    print "Trigram", "\t", "Collocations"
    for j in x:
       print j[0], "\t\t", j[1], "\t\t", j[2]

'''
Info class
To store all the relevant information of a corpus
'''
class Info:
    def __init__(self, wsb, wst):
        self.win_size_bgs=wsb
        self.win_size_tgs=wst
        wsb=str(wsb)
        wst=str(wst)
        self.params_t1_names = ["total_words", "num_ugs_c", "num_bgs_c",\
                                "num_tgs_c", "cov_90_ugs_c", "cov_80_bgs_c",\
                                "cov_70_tgs_c", "num_bgs_nc", "num_tgs_nc",\
                                "cov_80_bgs_nc", "cov_70_tgs_nc"]
        self.params_t1_nice_names = ["Total Words", "# Unigrams", "# Cont. Bigrams",\
                                "# Cont. Trigrams", "# Unigrams for 90% Coverage", \
                                "# Cont. Bigrams for 80% Coverage", "# Cont. Trigrams for 70% Coverage",\
                                "# Non Cont. Bigrams (WS="+wsb+")", "# Non Cont. Trigrams (WS="+wst+")",\
                                "# Non Cont. Bigrams (WS="+wsb+") for 80% Coverage",\
                                 "# Non Cont. Trigrams (WS="+wst+") for 70% Coverage"]
        self.params_t2_names = ["freq_ugs_c", "freq_bgs_c", "freq_tgs_c",\
                                "freq_bgs_nc", "freq_tgs_nc"]
        self.params_t2_nice_names = ["Frequency of Unigrams", "Frequency of Bigrams",\
                                     "Frequency of Trigrams", "Frequency of Non Cont. Bigrams (WS="+wsb+")",\
                                     "Frequency of Non Cont. Trigrams (WS="+wst+")"]
        self.params_t1_vals=dict()
        self.params_t2_vals=dict()
        self.params_t1_vals["total_words"]=None
        self.params_t1_vals["num_ugs_c"]=None
        self.params_t1_vals["num_bgs_c"]=None
        self.params_t1_vals["num_tgs_c"]=None
        self.params_t2_vals["freq_ugs_c"]=None
        self.params_t2_vals["freq_bgs_c"]=None
        self.params_t2_vals["freq_tgs_c"]=None
        self.params_t1_vals["cov_90_ugs_c"]=None
        self.params_t1_vals["cov_80_bgs_c"]=None
        self.params_t1_vals["cov_70_tgs_c"]=None
        self.params_t1_vals["num_bgs_nc"]=None
        self.params_t1_vals["num_tgs_nc"]=None
        self.params_t2_vals["freq_bgs_nc"]=None
        self.params_t2_vals["freq_tgs_nc"]=None
        self.params_t1_vals["cov_80_bgs_nc"]=None
        self.params_t1_vals["cov_70_tgs_nc"]=None


'''
Analysis of a set of words from a corpus
words: list of words (tokens) from corpus
todo_nc: to perform the non contigous collocation or not
num_to_show: number of examples/collocations to show
win_size_bgs: window size for non contigous collocation bigrams
win_size_tgs: window_size for non contigous collocation trigrams
'''
def analyse(words, todo_nc=True, num_to_show=20, win_size_bgs = 25, win_size_tgs = 10):
    my_info=Info(win_size_bgs, win_size_tgs)
    my_info.params_t1_vals["total_words"] = len(words)
    
    # unigrams are nothing but single tokens
    ugs = words
    # get a dictionary of word -> frequency
    fdist_ugs = nltk.FreqDist(ugs)
    # sort the dictionary based on frequency
    sorted_ugs = sorted(fdist_ugs.items(), key=operator.itemgetter(1), reverse=True)
    # collect all frequencies
    freq_ugs = [n[-1] for n in sorted_ugs]
    print "Number of unigrams: %d" % (len(sorted_ugs))
    print "Top %d unigrams in monotonically decreasing order of frequencies: " % (num_to_show)
    prettyprintugs(sorted_ugs[:num_to_show])
    printSep()
    # store relevant information
    my_info.params_t1_vals["num_ugs_c"] = len(sorted_ugs)
    my_info.params_t2_vals["freq_ugs_c"] = sorted_ugs
    
    # get all contigous bigrams
    finder_bgs = nltk.collocations.BigramCollocationFinder.from_words(words, 2)
    # get a dictionary of word -> frequency
    fdist_bgs = finder_bgs.ngram_fd.viewitems()
    # sort the dictionary based on frequency
    sorted_bgs = sorted(fdist_bgs, key=operator.itemgetter(1), reverse=True)
    # collect all frequencies
    freq_bgs = [n[-1] for n in sorted_bgs]
    print "Number of Cont. bigrams: %d" % (len(sorted_bgs))
    print "Top %d Cont. bigrams in monotonically decreasing order of frequencies: " % (num_to_show)
    prettyprintbgs(sorted_bgs[:num_to_show])
    printSep()
    # store relevant information
    my_info.params_t1_vals["num_bgs_c"] = len(sorted_bgs)
    my_info.params_t2_vals["freq_bgs_c"] = sorted_bgs
    
    # get bigram collocation measure containing different tests for finding collocations
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    # get top num_to_show collocations using Mutual Information Measure
    top_bgs_c_colloc = finder_bgs.nbest(bigram_measures.pmi, num_to_show)
    print "Top %d Cont. bigrams collocations using Mutual Information measure: " % (num_to_show)
    prettyprintbgs_col(top_bgs_c_colloc)
    printSep()
    '''
    Note that the code below follows same pattern as above (So no need for redundant comments)
    '''
    # get all non contigous collocations window size win_size_bgs
    finder_bgs_nc = nltk.collocations.BigramCollocationFinder.from_words(words, win_size_bgs)
    fdist_bgs_nc = finder_bgs_nc.ngram_fd.viewitems()
    sorted_bgs_nc = sorted(fdist_bgs_nc, key=operator.itemgetter(1), reverse=True)
    freq_bgs_nc = [n[-1] for n in sorted_bgs_nc]
    print "Number of Non Cont. bigrams (Window size = %d): %d" % (win_size_bgs, len(sorted_bgs_nc))
    print "Top %d Non Cont. bigrams (Window size = %d) in monotonically decreasing order of frequencies: " % (num_to_show, win_size_bgs)
    prettyprintbgs(sorted_bgs_nc[:num_to_show])
    printSep()
    my_info.params_t1_vals["num_bgs_nc"] = len(sorted_bgs_nc)
    my_info.params_t2_vals["freq_bgs_nc"] = sorted_bgs_nc
    my_info.win_size_bgs = win_size_bgs
    
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    top_bgs_nc_colloc = finder_bgs_nc.nbest(bigram_measures.pmi, num_to_show)
    print "Top %d Non Cont. bigrams collocations (Window size = %d) in monotonically decreasing order of frequencies: " % (num_to_show, win_size_bgs)
    prettyprintbgs_col(top_bgs_nc_colloc)
    printSep()

    finder_tgs = nltk.collocations.TrigramCollocationFinder.from_words(words)
    fdist_tgs = finder_tgs.ngram_fd.viewitems()
    sorted_tgs = sorted(fdist_tgs, key=operator.itemgetter(1), reverse=True)
    freq_tgs = [n[-1] for n in sorted_tgs]
    print "Number of Cont. trigrams: %d" % (len(sorted_tgs))
    print "Top %d Cont. trigrams in monotonically decreasing order of frequencies: " % (num_to_show)
    prettyprinttgs(sorted_tgs[:num_to_show])
    printSep()
    my_info.params_t1_vals["num_tgs_c"] = len(sorted_tgs)
    my_info.params_t2_vals["freq_tgs_c"] = sorted_tgs
    
    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    top_tgs_c_colloc = finder_tgs.nbest(trigram_measures.pmi, num_to_show)
    print "Top %d Cont. trigrams collocations using Mutual Information measure: " % (num_to_show)
    prettyprinttgs_col(top_tgs_c_colloc)
    printSep()
    
    finder_tgs_nc = nltk.collocations.TrigramCollocationFinder.from_words(words, win_size_tgs)
    fdist_tgs_nc = finder_tgs_nc.ngram_fd.viewitems()
    sorted_tgs_nc = sorted(fdist_tgs_nc, key=operator.itemgetter(1), reverse=True)
    freq_tgs_nc = [n[-1] for n in sorted_tgs_nc]
    print "Number of Non Cont. trigrams (Window size = %d): %d" % (win_size_tgs, len(sorted_tgs_nc))
    print "Top %d Non Cont. trigrams (Window size = %d) in monotonically decreasing order of frequencies: " % (num_to_show, win_size_tgs)
    prettyprinttgs(sorted_tgs_nc[:num_to_show])
    printSep()
    my_info.params_t1_vals["num_tgs_nc"] = len(sorted_tgs_nc)
    my_info.params_t2_vals["freq_tgs_nc"] = sorted_tgs_nc
    my_info.win_size_tgs = win_size_tgs

    trigram_measures = nltk.collocations.TrigramAssocMeasures()
    top_tgs_c_colloc = finder_tgs.nbest(trigram_measures.pmi, num_to_show)
    print "Top %d Non Cont. trigrams collocations (Window size = %d) in monotonically decreasing order of frequencies: " % (num_to_show, win_size_tgs)
    prettyprinttgs_col(top_tgs_c_colloc)
    printSep()
    printSep2()


    total_words = len(words)
    # req proportion of total coverage by unigrams
    pugs=0.9
    cum_freq_ugs = np.cumsum(freq_ugs)-1
    print "Number of most frequent words required for", pugs*100, "% coverage:", \
            (np.sum(cum_freq_ugs<np.round(pugs*total_words))+1), "out of", len(sorted_ugs)
    printSep()
    #my_info.params_t1_vals["cov_90_ugs_c"] =  ((np.sum(cum_freq_ugs<np.round(pugs*total_words))+1), len(sorted_ugs))
    my_info.params_t1_vals["cov_90_ugs_c"] =  (np.sum(cum_freq_ugs<np.round(pugs*total_words))+1)

    pbgs=0.8
    cum_freq_bgs = np.cumsum(freq_bgs)-1
    print "Number of most frequent Cont. bigrams required for", pbgs*100, "% coverage:", \
            (np.sum(cum_freq_bgs<np.round(pbgs*np.sum(freq_bgs)))+1), "out of", len(sorted_bgs)
    printSep()
    #my_info.params_t1_vals["cov_80_bgs_c"] = ((np.sum(cum_freq_bgs<np.round(pbgs*np.sum(freq_bgs)))+1), len(sorted_bgs))
    my_info.params_t1_vals["cov_80_bgs_c"] = (np.sum(cum_freq_bgs<np.round(pbgs*np.sum(freq_bgs)))+1)
    
    if todo_nc:
        pbgs=0.8
        cum_freq_bgs_nc = np.cumsum(freq_bgs_nc)-1
        print "Number of most frequent Non. Cont. bigrams (Window size =", win_size_bgs , ") required for", pbgs*100, "% coverage:", \
                (np.sum(cum_freq_bgs_nc<np.round(pbgs*np.sum(freq_bgs_nc)))+1), "out of", len(sorted_bgs_nc)
        printSep()
        #my_info.params_t1_vals["cov_80_bgs_nc"] = ((np.sum(cum_freq_bgs_nc<np.round(pbgs*np.sum(freq_bgs_nc)))+1), len(sorted_bgs_nc))
        my_info.params_t1_vals["cov_80_bgs_nc"] = (np.sum(cum_freq_bgs_nc<np.round(pbgs*np.sum(freq_bgs_nc)))+1)
        

    ptgs=0.7
    cum_freq_tgs = np.cumsum(freq_tgs)-1
    print "Number of most frequent Cont. trigrams required for", ptgs*100, "% coverage:", \
            (np.sum(cum_freq_tgs<np.round(ptgs*np.sum(freq_tgs)))+1), "out of", len(sorted_tgs)
    printSep()
    #my_info.params_t1_vals["cov_80_tgs_c"] = ((np.sum(cum_freq_tgs<np.round(pbgs*np.sum(freq_tgs)))+1), len(sorted_tgs))
    my_info.params_t1_vals["cov_70_tgs_c"] = (np.sum(cum_freq_tgs<np.round(pbgs*np.sum(freq_tgs)))+1)
    
    if todo_nc:
        ptgs=0.7
        cum_freq_tgs_nc = np.cumsum(freq_tgs_nc)-1
        print "Number of most frequent Non. Cont. trigrams (Window size =", win_size_tgs , ") required for", ptgs*100, "% coverage:", \
                (np.sum(cum_freq_tgs_nc<np.round(ptgs*np.sum(freq_tgs_nc)))+1), "out of", len(sorted_tgs_nc)
        printSep()
        #my_info.params_t1_vals["cov_80_tgs_nc"] = ((np.sum(cum_freq_tgs_nc<np.round(pbgs*np.sum(freq_tgs_nc)))+1), len(sorted_tgs_nc))
        my_info.params_t1_vals["cov_70_tgs_nc"] = (np.sum(cum_freq_tgs_nc<np.round(pbgs*np.sum(freq_tgs_nc)))+1)
    
    # return all relevant information collected
    return my_info

'''
Rule based word segmenter
sent: a string (sentence)
word_delimiters: delimiters used for tokenization
First remove all white spaces using split
Then check if any word ends with a delimiters, hence tokenize
'''
def heuristic_word_segmenter(sent, word_delimiters):
    words=[]
    for w in sent.split():
        if str(w[-1]) in word_delimiters:
            if len(w[:-1]) > 0:
                words.append(w[:-1])
                # don't loose the delimiter
                words.append(str(w[-1]))
        else:
            words.append(w)
    return words

'''
Putative sentence segmenter
text: raw text of corpus
sent_delimiters: delimiters used for tokenization
Accumulate till a delimiter is seen
push into list as delimiter is seen with delimiter
'''
def putative_sent_segmenter(text, sent_delimiters = [".", "?", "!", "-", ";", ":"]):
    putative_sents = []
    accum = ""
    for i in range(len(text)):
        accum += str(text[i])
        if text[i] in sent_delimiters:
            putative_sents.append(accum)
            # reset accumulator
            accum=""
    if accum != "":
        putative_sents.append(accum)
    return putative_sents

'''
Sentence merger v1
putative_sents: output by putative_sent_segmenter
sent_end_mergers: text/pattern used for merging (like dr, mr etc for Dr. Mr.)
Looks for periods and checks if the text preeceding is in sent_end_mergers
'''
def sent_merger_v1(putative_sents, sent_end_mergers):
    merged_putative_sents = []
    i = 0
    while i < (len(putative_sents)-1):
        fstr = ""
        prev_i = i
        for j in range(i, len(putative_sents)-1):
            s1 = putative_sents[j]
            # if last delimiter is not period then break
            if s1[-1]!='.':
                break
            
            # get the toke just before period
            w_tok_j = s1.lower().split()[-1]
            w_tok_j = w_tok_j.split(".")[0]
            
            # check if token is in sent_end_mergers
            if w_tok_j in sent_end_mergers:
                fstr += s1
                i += 1
            else:
                break
        fstr += putative_sents[i]
        merged_putative_sents.append(fstr)
        i += 1
    merged_putative_sents.append(putative_sents[-1])
    return merged_putative_sents

'''
Sentence merger v2
putative_sents_2: output by sent_merger_v1
sent_delimiters_2: used for segmenting
checks if there is a lower case letter after ? or a naem after !
if yes then merge the segmented sentences
'''
def sent_merger_v2(putative_sents_2, sent_delimiters_2 = ["?", "!"]):
    return putative_sents_2
    merged_putative_sents_final = []
    k = 0
    while k < (len(putative_sents_2)):
        putative_sents = putative_sent_segmenter(putative_sents_2[k], sent_delimiters_2)
        i = 0
        merged_putative_sents = []
        while i < (len(putative_sents)-1):
            fstr = ""
            prev_i = i
            for j in range(i, len(putative_sents)-1):
                s1 = putative_sents[j]
                w_tok_j = heuristic_word_segmenter(s1)
                my_word = w_tok_j[-1]
                my_word = str(my_word)
                
                tagged_word = pos_tag([my_word])
                wrd, my_tag = tagged_word[0]
                if len(my_word)>0 and my_word[0].islower():
                    fstr += s1
                    i += 1
                elif my_tag == "NNP":
                    fstr += s1
                    i += 1
                else:
                    break
            fstr += putative_sents[i]
            merged_putative_sents.append(fstr)
            i += 1
        merged_putative_sents.append(putative_sents[-1])
        merged_putative_sents_final += merged_putative_sents
        k = k+1
    return merged_putative_sents_final

'''
Rule Based Sentence Segmenter
text: raw corpus text
Uses putative_sent_segmenter, sent_merger_v1, sent_merger_v2
'''
def heuristic_sent_segmenter(text):
    sent_delimiters = [".", "?", "!", "-", ";", ":"]
    putative_sents = putative_sent_segmenter(text, sent_delimiters)
    print "Using putative delimiters: ", sent_delimiters
    print "Number of sentences after putting putative sentence boundaries: %d" % (len(putative_sents))
    
    sent_end_mergers = ["dr", "mr", "ms", "mrs", "vs"]
    merged_putative_sents = sent_merger_v1(putative_sents, sent_end_mergers)
    print "Using sentence mergers: ", sent_end_mergers
    print "Number of sentences after merging: %d" % (len(merged_putative_sents))
    
    sent_delimiters = [".", "-", ";", ":"]
    putative_sents = putative_sent_segmenter(text, sent_delimiters)
    merged_putative_sents = sent_merger_v1(putative_sents, sent_end_mergers)
    merged_putative_sents_final = sent_merger_v2(merged_putative_sents)
    print "Using sentence mergers: Lower case after ? and name after ! (by POS Tagging)"
    print "Number of sentences after merging: %d" % (len(merged_putative_sents_final))
    return merged_putative_sents_final

'''
Chi Square test for finding bigram collocations contigous as well as non contigous
all_words: all words from corpus in sequence
window_size: For non contigous collocations
critical_value: if chi-sq value is greater than this value then dont reject collocation
Refer Stanford coreNLP Notes for the derivation.
'''
def chi_squared_test(all_words, window_size=2, critical_value=3.841):
    duples_list = []
    # find all duples
    for i in range(len(all_words)-window_size):
        for j in range(i+1, i+window_size):
            duples_list.append((all_words[i], all_words[j]))
    duples_set = set(duples_list)
    duples_set = list(duples_set)
    
    duples_l_arr = np.asarray(duples_list)
    bgs_list = len(duples_set)*[None]
    # all the tokens in corpus
    N = len(all_words)*1.0
    cnt=0
    for i in range(len(duples_set)):
        # compute statistics required for computing chi-sq
        w1,w2 = duples_set[i]
        O11 = np.sum((duples_l_arr[:,0]==w1)*(duples_l_arr[:,1]==w2))*1.0
        O12 = np.sum((duples_l_arr[:,0]!=w1)*(duples_l_arr[:,1]==w2))*1.0
        O21 = np.sum((duples_l_arr[:,0]==w1)*(duples_l_arr[:,1]!=w2))*1.0
        O22 = np.sum((duples_l_arr[:,0]!=w1)*(duples_l_arr[:,1]!=w2))*1.0
        
        chi_sq = ((O11*O22-O12*O21))/((O11+O12)*(O11+O21))
        chi_sq = (chi_sq*(O11*O22-O12*O21))/((O22+O21)*(O22+O12))
        chi_sq = chi_sq * N
        if chi_sq < 0:
            print "overflow: Some Problem"
            sys.exit(1)
        if chi_sq >= critical_value:
            bgs_list[cnt]=(w1,w2,chi_sq)
            cnt+=1
    bgs_list=bgs_list[:cnt]
    # sort collocations in decreasing order of chi_sq value
    sorted_by_third = sorted(bgs_list, key=lambda tup: tup[2], reverse=True)
    return sorted_by_third

'''
Bar plot utility functions
'''
def barplot(x_vals, y_vals, title, ylab, xlab, fpath):
    font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 10}
    matplotlib.rc('font', **font)
    
    plt.figure(figsize=(10.,10.))
    x_pos = x_vals
    plt.bar(x_pos, y_vals, align='center', width=0.5, color='r', alpha=0.4)
    plt.xticks(x_pos, xlab)
    plt.ylabel(ylab)
    plt.title(title)
    plt.grid(True)
    plt.savefig(fpath)
    plt.close()

'''
Histogram plot utility function
'''
def histplot(x_list, nbins, title, xlab, ylab, legends, fpath):
    x_list = np.asarray(x_list)
    plt.figure(figsize=(10,10))
    #bins = np.linspace(np.min(x_list), np.max(x_list), nbins)
    bins = np.linspace(0, 15000, nbins)
    ftime=True
    for i in range(0,x_list.shape[0]):
        if ftime:
            plt.hist(x_list[i], bins, alpha=1./x_list.shape[0], label=legends[i])
            plt.xlabel(xlab)
            plt.ylabel(ylab)
            plt.title(title)
            ftime=False
        else:
            plt.hist(x_list[i], bins, alpha=1./x_list.shape[0], label=legends[i])
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(fpath)
    plt.close()

'''
Compare different Info
info_list: list of Info
names: Names assigned to each Info
hist_thresh: min frequency required to be plotted on histogram
hist_bins: Number of histogram bins
'''
def compare(info_list, names, hist_thresh=5, hist_bins=100, hist_thresh_up=1000):
    t1_names=info_list[0].params_t1_names
    t2_names=info_list[1].params_t2_names
    t1_nice_names=info_list[0].params_t1_nice_names
    t2_nice_names = info_list[0].params_t2_nice_names
    for i in range(len(t1_names)):    
        x_vals=[]
        main_str=" "+t1_nice_names[i]+" "
        fpath="plots/"+t1_names[i]+".png"
        xlab="Number of"+main_str+"detected"
        title="Comparing different approaches for"+main_str
        check=False
        for k in info_list:
            if k.params_t1_vals[t1_names[i]] is None:
                check=True
                break
            x_vals.append(k.params_t1_vals[t1_names[i]])
        if check:
            y_vals=np.arange(len(names)-1)
            ylab=names[:-1]
        else:
            y_vals=np.arange(len(names))
            ylab=names
        barplot(y_vals, x_vals, title, xlab, ylab, fpath)
    '''
    for i in range(len(t2_names)):
        x_list=[]
        ylab="log log frequency"
        xlab="Un/B/Trigrams (see title)"
        main_str=" "+t2_nice_names[i]+" "
        title="Histogram of"+main_str
        fpath="plots/"+t2_names[i]+".png"
        check=False
        for k in info_list:
            if k.params_t2_vals[t2_names[i]] is None:
                check=True
                break
            freq = [n[-1] for n in k.params_t2_vals[t2_names[i]]]
            freq = np.asarray(freq)
            freq2 = freq[((freq>hist_thresh)*(freq<hist_thresh_up))]
            freq2 = np.log(np.log(freq2))
            freq = freq2.tolist()
            x_list.append(freq)
        nms=names
        if check:
            nms=names[:-1]
        histplot(x_list, hist_bins, title, xlab, ylab, names, fpath
    '''

'''
Lemmatization
words_list: list of words to be lemmatized
'''
def lemmatization(words_list):
    wordnet_lemmatizer = WordNetLemmatizer()
    words_list = [wordnet_lemmatizer.lemmatize(w) for w in words_list]
    all_dict = set(words_list)      
    print "Number of words in dictionary: %d" % (len(all_dict))
    printSep()
    return words_list   

'''
Get list of words from text from a corpus
text: text from a corpus
Uses sentence segmentation and then word segmentation from nltk
'''
def getListOfWords(text):
    sent_tokenize_list = sent_tokenize(text)
    print "Number of sentences after sent_tokenize: %d" % (len(sent_tokenize_list))
    print "Sample sentence:"
    print sent_tokenize_list[10]
    printSep()
    N=len(sent_tokenize_list)
    words_list=N*[None]
    ind=0
    for sent in sent_tokenize_list:
        w = word_tokenize(sent)
        words_list[ind]=w
        ind+=1
    words_list_final=[]
    for w in words_list:
        words_list_final += w
    
    all_dict = set(words_list_final)      
    print "Number of words in dictionary: %d" % (len(all_dict))
    printSep()
    return words_list_final

'''
Get list of words from text from a corpus
text: text from a corpus
Uses Heuristic based sentence segmentation and then word segmentation
'''
def getListOfWordsHeuristicBased(text):
    h_sents = heuristic_sent_segmenter(text)
    h_words = []
    for sent in h_sents:
        w = heuristic_word_segmenter(sent, [".", "?", "!", "-", ";", ":", ",", "'", "\"", "`", "`", "$", "+"])
        h_words += w
    all_dict = set(h_words)      
    print "Number of words in dictionary: %d" % (len(all_dict))
    printSep()
    return h_words

'''
Utility function to creat Info from coreNLP output
fpath: path to text file of coreNLP
'''
def readCoreNLPInfo(fpath):
    f=open(fpath, "r")
    content=f.read()
    f.close();
    content=content.split()
    my_info=Info(0,0)
    
    ind=0
    my_info.params_t1_vals["total_words"] = int(content[ind])
    ind+=1
    my_info.params_t1_vals["num_ugs_c"] = int(content[ind])
    ind+=1
    x=[]
    for i in range(my_info.params_t1_vals["num_ugs_c"]):
        ind+=1
        x.append(int(content[ind]))
        ind+=1
    my_info.params_t1_vals["freq_ugs_c"] = x
    my_info.params_t1_vals["cov_90_ugs_c"] = int(content[ind])
    ind+=1
    
    my_info.params_t1_vals["num_bgs_c"] = int(content[ind])
    ind+=1
    x=[]
    for i in range(my_info.params_t1_vals["num_bgs_c"]):
        ind+=2
        x.append(int(content[ind]))
        ind+=1
    my_info.params_t1_vals["freq_bgs_c"] = x
    my_info.params_t1_vals["cov_80_bgs_c"] = int(content[ind])
    ind+=1
    
    my_info.params_t1_vals["num_tgs_c"] = int(content[ind])
    ind+=1
    x=[]
    for i in range(my_info.params_t1_vals["num_tgs_c"]):
        ind+=3
        x.append(int(content[ind]))
        ind+=1
    my_info.params_t1_vals["freq_tgs_c"] = x
    my_info.params_t1_vals["cov_70_tgs_c"] = int(content[ind])
    ind+=1
    
    my_info1 = my_info
    my_info=Info(0,0)
    
    my_info.params_t1_vals["total_words"] = int(content[ind])
    ind+=1
    my_info.params_t1_vals["num_ugs_c"] = int(content[ind])
    ind+=1
    x=[]
    for i in range(my_info.params_t1_vals["num_ugs_c"]):
        ind+=1
        x.append(int(content[ind]))
        ind+=1
    my_info.params_t1_vals["freq_ugs_c"] = x
    my_info.params_t1_vals["cov_90_ugs_c"] = int(content[ind])
    ind+=1
    
    my_info.params_t1_vals["num_bgs_c"] = int(content[ind])
    ind+=1
    x=[]
    for i in range(my_info.params_t1_vals["num_bgs_c"]):
        ind+=2
        x.append(int(content[ind]))
        ind+=1
    my_info.params_t1_vals["freq_bgs_c"] = x
    my_info.params_t1_vals["cov_80_bgs_c"] = int(content[ind])
    ind+=1
    
    my_info.params_t1_vals["num_tgs_c"] = int(content[ind])
    ind+=1
    x=[]
    for i in range(my_info.params_t1_vals["num_tgs_c"]):
        ind+=3
        x.append(int(content[ind]))
        ind+=1
    my_info.params_t1_vals["freq_tgs_c"] = x
    my_info.params_t1_vals["cov_70_tgs_c"] = int(content[ind])
    ind+=1
    return my_info1, my_info
    

'''
Run the analyses using nltk, lemmatization+nltk, heuristic, lemmatization+heuristic
'''
def run():
    print "1.1.Loading 'austen-emma.txt' corpus from nltk toolkit"
    emma = nltk.corpus.gutenberg.raw('austen-emma.txt')
    
    printSep2()
    print "NLTK"
    printSep2()
    words = getListOfWords(emma)
    info_nltk=analyse(words)
    
    printSep2()
    print "Lemmatization + NLTK"
    printSep2()
    words = lemmatization(words)
    info_lemmatized=analyse(words)
    
    printSep2()
    print "Heuritic Based"
    printSep2()
    h_words = getListOfWordsHeuristicBased(emma)
    info_heuristic=analyse(h_words)
    
    printSep2()
    print "Lemmatization + Heuristic"
    printSep2()
    h_words = lemmatization(h_words)
    info_lemmatized2=analyse(h_words)
    
    printSep2()
    print "Reading coreNL Info"
    printSep2()
    info_corenlp, info_corenlp_lemmatiized=readCoreNLPInfo("data/out2.txt")
    '''
    print "Compare"
    hist_thresh=10
    hist_bins=100
    hist_thresh_up=10000000
    while hist_thresh!=0:
        #hist_thresh_up=int(raw_input("Enter threshold up:"))
        #hist_thresh=int(raw_input("Enter threshold: "))
        #hist_bins=int(raw_input("Enter num bins: "))
        sys.stdout.flush()
        if hist_thresh==0:
            break
        compare([info_nltk, info_lemmatized, info_heuristic, info_lemmatized2, info_corenlp, info_corenlp_lemmatiized], \
                ["NLTK", "Lemmatized+NLTK", "Heuristic", "Lemmatized+Heuristic", "coreNLP", "Lemmatized+coreNLP"], hist_thresh, hist_bins, hist_thresh_up)
        break
    '''
    print "Chi square test for cont. and non-cont. bigrams collocations finding"
    win_size = 2
    num_to_show = 20
    bgs_list = chi_squared_test(words, win_size)
    print "Number of Bigram Collocations (Window size = %d) with P.Chi Squared Test: %d" % (win_size, len(bgs_list))
    print "Top %d Non Cont. bigrams (Window size = %d) in monotonically decreasing order of frequencies: " % (num_to_show, win_size)
    prettyprintbgs_col(bgs_list[:num_to_show])
    
    win_size = 25
    bgs_list = chi_squared_test(words, win_size)
    print "Number of Bigram Collocations (Window size = %d) with P.Chi Squared Test: %d" % (win_size, len(bgs_list))
    print "Top %d Non Cont. bigrams (Window size = %d) in monotonically decreasing order of frequencies: " % (num_to_show, win_size)
    prettyprintbgs_col(bgs_list[:num_to_show])
    

'''
Run the analyses for bonus question on pubmed_1
'''
def runBonus():
    # Bonus Problem
    printSep2()
    num_to_show = 20
    print "Bonus Problem"
    f = open("pubmed_1", "rb")
    content = f.read()
    f.close()
    content = content.decode('utf-8')
    content = content[:1000000]
    words = getListOfWords(content)
    #words = words[:200000]
    ugs = words
    fdist_ugs = nltk.FreqDist(ugs)
    #sorted_ugs = sorted(fdist_ugs.items(), key=operator.itemgetter(1), reverse=True)
    freq_ugs = [value for key,value in fdist_ugs.iteritems()]
    print "Number of unigrams: %d" % (len(freq_ugs))
    freq = np.asarray(freq_ugs)
    hist_thresh=10
    nbins=100
    hist_thresh_up=1000
    while hist_thresh!=0:
        
        #nbins = int(raw_input("Enter number of bins: "))
        #hist_thresh=int(raw_input("Enter hist threshold: "))
        #hist_thresh_up=int(raw_input("Enter hist threshold up: "))
        #totakeloglog=int(raw_input("To take log log:?"))
        totakeloglog=1
        if hist_thresh==0:
            break
        freq2 = freq[((freq>hist_thresh)*(freq<hist_thresh_up))]
        if totakeloglog:
            freq2 = np.log(np.log(freq2))
        freq2 = freq2.tolist()
        x_list = [freq2]
        title = "Histogram of word frequency distribtuin pubmed_1"
        xlab = ""
        ylab = "log log frequency"
        legends = ["pubmed_1"]
        fpath = "plots/pubmed_1.png"
        histplot(x_list, nbins, title, xlab, ylab, legends, fpath)
        break
    
    finder_bgs = nltk.collocations.BigramCollocationFinder.from_words(words)
    fdist_bgs = finder_bgs.ngram_fd.viewitems()
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    top_bgs_c_colloc = finder_bgs.nbest(bigram_measures.pmi, len(fdist_bgs))
    
    f = open("data/bonus_colloc.txt", "w")
    print "Top %d Cont. bigrams collocations using Mutual Information measure: " % (num_to_show)
    for k in top_bgs_c_colloc:
        f.write((k[0]+" "+k[1]+"\n").encode('utf-8'))
    f.close()
    print "All collocations in decreasing order printed in bonus_colloc.txt"
    

if __name__=="__main__":
    run()
    runBonus()