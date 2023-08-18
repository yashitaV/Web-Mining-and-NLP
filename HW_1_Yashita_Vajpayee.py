#!/usr/bin/env python
# coding: utf-8

# # <center>HW #1: Analyze Documents by Numpy</center>

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. </div>

# **Instructions**: 
# - Please read the problem description carefully
# - Make sure to complete all requirements (shown as bullets) . In general, it would be much easier if you complete the requirements in the order as shown in the problem description
# - Follow the Submission Instruction to submit your assignment

# **Problem Description**
# 
# In this assignment, you'll write functions to analyze an article to find out the word distributions and key concepts. 
# 
# The packages you'll need for this assignment include `numpy` and `string`. Some useful functions:
# - string, list, dictionary: `split`, `count`, `index`,`strip`
# - numpy: `sum`, `where`,`log`, `argsort`,`argmin`, `argmax` 

# ## Q1. Define a function to analyze word counts in an input sentence
# 
# 
# Define a function named `tokenize(doc)` which does the following: 
# 
# * accepts a document (i.e., `doc` parameter) as an input
# * first splits a document into paragraphs by delimiter `\n\n` (i.e. two new lines)
# * for each paragraph, 
#     - splits it into a list of tokens by **space** (including tab, and new line). 
#         - e.g., `it's a hello world!!!` will be split into tokens `["it's", "a","hello","world!!!"]`  
#     - removes the **leading/trailing punctuations or spaces** of each token, if any 
#         - e.g., `world!!! -> world`, while `it's` does not change
#         - hint, you can import module *string*, use `string.punctuation` to get a list of punctuations (say `puncts`), and then use function `strip(puncts)` to remove leading or trailing punctuations in each token
#     - a token has at least two characters  
#     - converts all tokens into lower case 
#     - find the count of each unique token and save the count as a dictionary, named `word_dict`, i.e., `{world: 1, a: 1, ...}` 
# * creates another dictionary,say `para_dict`, where a key is the order of each paragraph in `doc`, and the value is the `word_dict` generated from this paragraph
# * returns the dictionary `para_dict` and the paragraphs in the document
#     

# In[44]:


import string
from string import punctuation
import pandas as pd
import numpy as np

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[45]:


def tokenize(doc):
    
    para_dict, para = None, None
    para = doc.split("\n\n")  ## Splitting into paragraphs 
    para_dict = dict()
    for i, paras in enumerate(para):
        tokens = paras.split()    ## Spliting into tokens
        word_dict = dict()
        for word in tokens:
            word = word.strip(string.punctuation).lower()# Stripping punctuations and converting into lower case
            if len(word) >= 2:   ## token has at least two characters, counting word frequeny
                if word in word_dict:
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
        para_dict[i] = word_dict    ## Creating paragraph dictionary with word_dict as value
   
    
    return para_dict, para


# In[46]:


# test your code
doc = "it's a hello world!!!\nit is hello world again.\n\nThis is paragraph 2."
print(doc)
tokenize(doc)


# ## Q2. Generate a document term matrix (DTM) as a numpy array
# 
# 
# Define a function `get_dtm(doc)` as follows:
# - accepts a document, i.e., `doc`, as an input
# - uses `tokenize` function you defined in Q1 to get the word dictionary for each paragraph in the document 
# - pools the keys from all the word dictionaries to get a list of  unique words, denoted as `unique_words` 
# - creates a numpy array, say `dtm` with a shape (# of paragraphs x # of unique words), and set the initial values to 0. 
# - fills cell `dtm[i,j]` with the count of the `j`th word in the `i`th paragraph 
# - returns `dtm` and `unique_words`

# In[47]:


def get_dtm(doc):
    
    dtm, all_words = None, None
    
    para_dict, lines = tokenize(doc)  ## getting  word dictionary for each paragraph using tokenize function
    unique_words = []       ## list for unique words
    for para in para_dict.values():
        for word in para.keys():
            if word not in unique_words:   
                unique_words.append(word)  ## appending the list if unique word
    dtm = np.zeros((len(lines), len(unique_words)))  ## Creating dtm matrix according to mentioned dimensions
    for i, para in enumerate(para_dict.values()):
        for j, word in enumerate(unique_words):
            if word in para:
                dtm[i,j] = para[word]    ## Filling in dtm matrix
    all_words = unique_words
    
    return dtm, all_words


# In[48]:


dtm, all_words = get_dtm(doc)
dtm
all_words


# In[49]:


# A test document. This document can be found at https://www.wboi.org/npr-news/2023-01-26/everybody-is-cheating-why-this-teacher-has-adopted-an-open-chatgpt-policy

doc = open('chatgpt_npr.txt').read()
dtm, all_words = get_dtm(doc)


# In[50]:


print(doc)


# In[51]:


# To ensure dtm is correct, check what words in a paragraph have been captured by dtm

p = 6 # paragraph id

[w for i,w in enumerate(all_words) if dtm[p][i]>0] 


# ## Q3 Analyze DTM Array 
# 
# 
# **Don't use any loop in this task**. You should use array operations to take the advantage of high performance computing.
# 

# Define a function named `analyze_dtm(dtm, words, paragraphs)` which:
# * takes an array $dtm$ and $words$ as an input, where $dtm$ is the array you get in Q2 with a shape $(m \times n)$, and $words$ contains an array of words corresponding to the columns of $dtm$.
# * calculates the paragraph frequency for each word $j$, e.g. how many paragraphs contain word $j$. Save the result to array $df$. $df$ has shape of $(n,)$ or $(1, n)$. 
# * normalizes the word count per paragraph: divides word count, i.e., $dtm_{i,j}$, by the total number of words in paragraph $i$. Save the result as an array named $tf$. $tf$ has shape of $(m,n)$. 
# * for each $dtm_{i,j}$, calculates $tfidf_{i,j} = \frac{tf_{i, j}}{1+log(df_j)}$, i.e., divide each normalized word count by the log of the paragraph frequency of the word (add 1 to the denominator to avoid dividing by 0).  $tfidf$ has shape of $(m,n)$ 
# * prints out the following (hint: you can zip words and their values into a list so that there is no need for loop during printing):
#     
#     - the total number of words in the document represented by $dtm$ 
#     - the number of paragraphs and the number of unique words in the document
#     - the most frequent top 10 words in this document    
#     - top-10 words that show in most of the paragraphs, i.e. words with the top 10 largest $df$ values (show words and their $df$ values) 
#     - the shortest paragraph (i.e., the one with the least number of words) 
#     - top-5 words with the largest $tfidf$ values in the longest paragraph (show words and values) 
# 
# Note, for all the steps, **do not use any loop**. Just use array functions and broadcasting for high performance computation.
# 
# Your answer may be different from the example output, since words may have the same values in the dtm but are kept in positions

# In[52]:


import numpy as np

def analyze_dtm(dtm, words, paragraphs):
    m, n = dtm.shape
    df = np.count_nonzero(dtm, axis=0)
    tf = dtm / np.sum(dtm, axis=1, keepdims=True)   ## Term frequency
    tfd_idf = tf * np.log(m / (df + 1))    ## tf-idf formula

    print(f"\nTotal number of words:\n {np.sum(dtm)}")
    
    print(f"\nNumber of paragraphs: {m}, Number of unique words:\n {n}")
    
    print("\nMost frequent top 10 words:\n")
    print(sorted(zip(words, np.sum(dtm, axis=0)), key=lambda x: x[1], reverse=True)[:10]) ## lambda function to get top 10 values
    
    print("\nThe top 10 words with highest df values:\n")
    print(sorted(zip(words, df), key=lambda x: x[1], reverse=True)[:10])
    min_idx = np.argmin(np.sum(dtm, axis=1))  ## using argmin to get minimum index
    
    print(f"\nThe shortest paragraph :\n {paragraphs[min_idx]}")
    max_idx = np.argmax(np.sum(dtm, axis=1))
    top_5 = np.argsort(tfd_idf[max_idx])[-5:]
    
    print("\nThe top 5 words with the highest tf-idf values in the longest paragraph:\n")
    print(list(zip(words[top_5], tfd_idf[max_idx][top_5])))


# In[53]:


para_dict, paragraphs = tokenize(doc)

# convert words in numpy arrays so that you can use array slicing
words = np.array(all_words)

analyze_dtm(dtm, words, paragraphs)


# ## Q4. Find co-occuring words (Bonus) 
# 
# Can you leverage $dtm$ array you generated to find what words frequently coocur with a specific word? For example, "students" and "chatgpt" coocur in 4 paragraphs. 
# 
# Define a function `find_coocur(w1, w2, dtm, words)`, which returns the paragraphs containong both words w1 and w2.
# 
# Use a pdf file to describe your ideas and also implement your ideas. Again, `DO NOT USE LOOP`! 
# 

# In[54]:


def find_coocur(w1, w2, dtm, words, paragraphs):
    w1_index = np.where(words == w1)[0][0]    ## Getting index of word 1
    w2_index = np.where(words == w2)[0][0]    ## Getting index of word 2
    cooccur_para_indexes = np.where((dtm[:, w1_index] != 0) & (dtm[:, w2_index] != 0))[0] ## Checking co-occur condition
    result = np.array([paragraphs[i] for i in cooccur_para_indexes])   ## printing paragraphs having co-occurences
    
    return result


# In[55]:


ps = find_coocur('chatgpt','students',dtm, words, paragraphs)
len(ps)
ps


# **Put everything together and test using main block**

# In[56]:


# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":  
    
    # Test Question 1
    doc = "it's a hello world!!!\nit is hello world again.\n\nThis is paragraph 2."
    print("Test Question 1")
    para_dict, paragraphs = tokenize(doc)
    print(tokenize(doc))
    
    
    # Test Question 2
    print("\nTest Question 2")
    dtm, all_words = get_dtm(doc)

    print(dtm)
    print(all_words)
    
    
    #3 Test Question 3
    
    doc = open("chatgpt_npr.txt", 'r').read()
    
    para_dict, paragraphs = tokenize(doc)
    dtm, all_words = get_dtm(doc)

    print("\nTest Question 3")
    words = np.array(all_words)

    tfidf = analyze_dtm(dtm, words, paragraphs)
    
    

