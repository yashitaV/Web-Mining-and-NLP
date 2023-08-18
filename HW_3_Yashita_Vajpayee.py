#!/usr/bin/env python
# coding: utf-8

# # HW3: Natural Language Processing

#  <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. </div>

# ## Problem Description
# 
# In this assignment, we'll use what we learned in NLP module to compare ChatGPT-generated text with human-generated answers. A dataset with 200 questions and answers has been provided for you to use. The dataset can be found at https://huggingface.co/datasets/Hello-SimpleAI/HC3.
# 
# 
# Please follow the instruction below to do the assessment step by step and answer all analysis questions.
# 

# In[64]:


import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import warnings 
warnings.filterwarnings('ignore')


# In[21]:


data = pd.read_csv("/Users/yashitavajpayee/Documents/Semester II/BIA 660 - Web mining/qa.csv")
data.head()


# ## Q1. Tokenize function
# 
# Define a function `tokenize(docs, lemmatized = True, remove_stopword = True, remove_punct = True)`  as follows:
#    - Take three parameters: 
#        - `docs`: a list of documents (e.g. questions)
#        - `lemmatized`: an optional boolean parameter to indicate if tokens are lemmatized. The default value is True (i.e. tokens are lemmatized).
#        - `remove_stopword`: an optional bookean parameter to remove stop words. The default value is True (i.e. remove stop words). 
#    - Split each input document into unigrams and also clean up tokens as follows:
#        - if `lemmatized` is turned on, lemmatize all unigrams.
#        - if `remove_stopword` is set to True, remove all stop words.
#        - if `remove_punct` is set to True, remove all punctuation tokens.
#        - remove all empty tokens and lowercase all the tokens.
#    - Return the list of tokens obtained for each document after all the processing. 
#    
# (Hint: you can use spacy package for this task. For reference, check https://spacy.io/api/token#attributes)

# In[22]:


import spacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp = spacy.load('en_core_web_sm')

def tokenize(docs, lemmatized=True, remove_stopword=True, remove_punct=True):
    tokenized_docs = []
    punct_chars = set(['.', ',', ';', ':', '!', '?', '-', '(', ')', '[', ']', '{', '}', '\'', '\"', '/', '\\', '@', '#', '$', '%', '&', '*', '+', '=', '<', '>', '|', '^', '_', '`', '~'])
    for doc in docs:
        tokens = nlp(doc)
        tokens = [token.lemma_ if lemmatized else token.text for token in tokens]
        tokens = [token.lower() for token in tokens if token.strip() != ""]
        if remove_stopword:
            tokens = [token for token in tokens if token not in STOP_WORDS]
        if remove_punct:
            tokens = [token for token in tokens if token not in punct_chars]
        tokenized_docs.append(tokens)
    return tokenized_docs


# In[23]:


# For simplicity, We will test one document

print(data["question"].iloc[0] + "\n")

print(f"1.lemmatized=True, remove_stopword=False, remove_punct = True:\n {tokenize(data['question'].iloc[0:1], lemmatized=True, remove_stopword=False, remove_punct = True)}\n")

print(f"2.lemmatized=True, remove_stopword=True, remove_punct = True:\n {tokenize(data['question'].iloc[0:1], lemmatized=True, remove_stopword=True, remove_punct = True)}\n")

print(f"3.lemmatized=False, remove_stopword=False, remove_punct = True:\n {tokenize(data['question'].iloc[0:1], lemmatized=False, remove_stopword=False, remove_punct = True)}\n")

print(f"4.lemmatized=False, remove_stopword=False, remove_punct = False:\n {tokenize(data['question'].iloc[0:1], lemmatized=False, remove_stopword=False, remove_punct = False)}\n")


# Test your function with different parameter configuration and observe the differences in the resulting tokens.

# ## Q2. Sentiment Analysis
# 
# 
# Let's check if there is any difference in sentiment between ChatGPT-generated and human-generated answers.
# 
# 
# Define a function `compute_sentiment(generated, reference, pos, neg )` as follows:
# - take four parameters:
#     - `gen_tokens` is the tokenized ChatGPT-generated answers by the `tokenize` function in Q1.
#     - `ref_tokens` is the tokenized human answers by the `tokenize` function in Q1.
#     - `pos` (`neg`) is the list of positive (negative) words, which can be find in Canvas NLP module.
# - for each ChatGPT-generated or human answer, compute the sentiment as `(#pos - #neg )/(#pos + #neg)`, where `#pos`(`#neg`) is the number of positive (negative) words found in each answer. If an answer contains none of the positive or negative words, set the sentiment to 0.
# - return the sentiment of ChatGPT-generated and human answers as two columns of DataFrame.
# 
# 
# Analysis: 
# - Try different tokenization parameter configurations (lemmatized, remove_stopword, remove_punct), and observe how sentiment results change.
# - Do you think, in general, which tokenization configuration should be used? Why does this configuration make the most senese?
# - Do you think, overall, ChatGPT-generated answers are more posive or negative than human answers? Use data to support your conclusion.
# 

# In[39]:


def compute_sentiment(gen_tokens, ref_tokens, pos, neg):
    
    def count_sentiment(tokens):
        count_pos = sum(1 for t in tokens if t in pos)
        count_neg = sum(1 for t in tokens if t in neg)
        if count_pos + count_neg == 0:
            return 0
        return (count_pos - count_neg) / (count_pos + count_neg)
    
    gen_sentiments = [count_sentiment(tokens) for tokens in gen_tokens]
    ref_sentiments = [count_sentiment(tokens) for tokens in ref_tokens]
    
    result = pd.DataFrame({
        'generated_sentiment': gen_sentiments,
        'reference_sentiment': ref_sentiments
    })
    return result


# In[40]:


gen_tokens = tokenize(data["chatgpt_answer"], lemmatized=False, remove_stopword=False, remove_punct = False)
ref_tokens = tokenize(data["human_answer"], lemmatized=False, remove_stopword=False, remove_punct = False)


# In[41]:


pos = pd.read_csv("/Users/yashitavajpayee/Documents/Semester II/BIA 660 - Web mining/positive-words.txt", header = None)
pos.head()

neg = pd.read_csv("/Users/yashitavajpayee/Documents/Semester II/BIA 660 - Web mining/negative-words.txt", header = None)
neg.head()


# In[42]:


result = compute_sentiment(gen_tokens, 
                           ref_tokens, 
                           pos[0].values,
                           neg[0].values)
result.head()


# ## Q3: Performance Evaluation
# 
# 
# Next, we evaluate how accurate the ChatGPT-generated answers are, compared to the human-generated answers. One widely used method is to calculate the `precision` and `recall` of n-grams. For simplicity, we only calculate bigrams here. You can try unigram, trigram, or n-grams in the same way.
# 
# 
# Define a funtion `bigram_precision_recall(gen_tokens, ref_tokens)` as follows:
# - take two parameters:
#     - `gen_tokens` is the tokenized ChatGPT-generated answers by the `tokenize` function in Q1.
#     - `ref_tokens` is the tokenized human answers by the `tokenize` function in Q1.
# - generate bigrams from each tokenized document in `gen_tokens` and `ref_tokens`.
# - for each pair of ChatGPT-generated and human answers: 
#     - find the overlapping bigrams between them.
#     - compute `precision` as the number of overlapping bigrams divided by the total number of bigrams from the ChatGPT-generated answer. In other words, the bigram in the ChatGPT-generated answer is considered as a predicted value. The `precision` measures the percentage of correctly generated bigrams out of all the generated bigrams.
#     - compute `recall` as the number of overlapping bigrams divided by the total number of bigrams from the human answer. In other words, the `recall` measures the percentage of bigrams from the human answer can be successfully retrieved.
# - return the precision and recall for each pair of answers.
# 
# 
# Analysis: 
# - Try different tokenization parameter configurations (lemmatized, remove_stopword, remove_punct), and observe how precison and recall change.
# - Which tokenization configuration can render the highest average precision and recall scores across all questions?
# - Do you think, overall, ChatGPT is able to mimic human in answering these questions?
# 
# 

# In[116]:


from nltk import ngrams
def bigram_precision_recall(gen_tokens, ref_tokens):
    
    result = None
    data = []
    
    for gen, ref in zip(gen_tokens, ref_tokens):
        gen_bigrams = set(ngrams(gen, 2))
        ref_bigrams = set(ngrams(ref, 2))
        
        overlapping = gen_bigrams.intersection(ref_bigrams)
        precision = len(overlapping) / len(gen_bigrams) if len(gen_bigrams) > 0 else 0
        recall = len(overlapping) / len(ref_bigrams) if len(ref_bigrams) > 0 else 0
        
        data.append({'overlapping': overlapping, 'precision': precision, 'recall': recall})
        result = pd.DataFrame(data)
    return result


# In[33]:


result = bigram_precision_recall(gen_tokens, 
                                 ref_tokens)
result.head()


# ## Q4 Compute TF-IDF
# 
# Define a function `compute_tf_idf(tokenized_docs)` as follows: 
# - Take paramter `tokenized_docs`, i.e., a list of tokenized documents by `tokenize` function in Q1
# - Calculate tf_idf weights as shown in lecture notes (Hint: feel free to reuse the code segment in NLP Lecture Notes (II))
# - Return the smoothed normalized `tf_idf` array, where each row stands for a document and each column denotes a word. 

# In[74]:


from sklearn.feature_extraction.text import TfidfVectorizer

def compute_tfidf(tokenized_docs, smooth=True):
    
    smoothed_tf_idf = None
    # Join the tokenized documents back into strings
    
    docs = [' '.join(doc) for doc in tokenized_docs]
    # Compute the tf-idf weights using scikit-learn's TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tf_idf = vectorizer.fit_transform(docs)
    if smooth:
        # Normalize and smooth the tf-idf matrix
        smoothed_tf_idf = (tf_idf.toarray() + 1) / (tf_idf.sum(axis=1).reshape(-1, 1) + 1)
        return smoothed_tf_idf
    else:
        return tf_idf.toarray()


# Try different tokenization options to see how these options affect TFIDF matrix:

# In[75]:


# Test tfidf generation using questions

question_tokens = tokenize(data["question"], lemmatized=True, remove_stopword=False, remove_punct = True)
dtm = compute_tfidf(question_tokens)
print(f"1.lemmatized=True, remove_stopword=False, remove_punct = True:\n Shape: {dtm.shape}\n")

question_tokens = tokenize(data["question"], lemmatized=True, remove_stopword=True, remove_punct = True)
dtm = compute_tfidf(question_tokens)
print(f"2.lemmatized=True, remove_stopword=True, remove_punct = True:\n Shape: {dtm.shape}\n")

question_tokens = tokenize(data["question"], lemmatized=False, remove_stopword=False, remove_punct = True)
dtm = compute_tfidf(question_tokens)
print(f"3.lemmatized=False, remove_stopword=False, remove_punct = True:\n Shape: {dtm.shape}\n")

question_tokens = tokenize(data["question"], lemmatized=False, remove_stopword=False, remove_punct = False)
dtm = compute_tfidf(question_tokens)
print(f"4.lemmatized=False, remove_stopword=False, remove_punct = False:\n Shape: {dtm.shape}\n")


# ## Q5. Assess similarity. 
# 
# 
# Define a function `assess_similarity(question_tokens, gen_tokens, ref_tokens)`  as follows: 
# - Take three inputs:
#    - `question_tokens`: tokenized questions by `tokenize` function in Q1
#    - `gen_tokens`: tokenized ChatGPT-generated answers by `tokenize` function in Q1
#    - `ref_tokens`: tokenized human answers by `tokenize` function in Q1
# - Concatenate these three token lists into a single list to form a corpus
# - Calculate the smoothed normalized tf_idf matrix for the concatenated list using the `compute_tfidf` function defined in Q3.
# - Split the tf_idf matrix into sub-matrices corresponding to `question_tokens`, `gen_tokens`, and `ref_tokens` respectively
# - For each question, find its similarities to the paired ChatGPT-generated answer and human answer.
# - For each pair of ChatGPT-generated answer and human answer, find their similarity
# - Print out the following:
#     - the question which has the largest similarity to the ChatGPT-generated answer.
#     - the question which has the largest similarity to the human answer.
#     - the pair of ChatGPT-generated and human answers which have the largest similarity.
# - Return a DataFrame with the three columns for the similarities among questions and answers.
# 
# 
# 
# Analysis: 
# - Try different tokenization parameter configurations (lemmatized, remove_stopword, remove_punct), and observe how similarities change.
# - Based on similarity, do you think ChatGPT-generate answers are more (or less) relevant to questions than human answers?

# In[117]:


from sklearn.metrics.pairwise import cosine_similarity
def assess_similarity(question_tokens, gen_tokens, ref_tokens):
    """
    Computes the similarity scores between a question, a generated answer, and a set of reference answers.
    Returns a pandas DataFrame with the similarity scores.
    
    :param question_tokens: list of str, the tokenized question
    :param gen_tokens: list of str, the tokenized generated answer
    :param ref_tokens: list of list of str, the tokenized reference answers
    :return: pandas DataFrame, the similarity scores between the question, generated answer, and reference answers
    """
    
    # Concatenate all tokens into a single list
    concatenated_tokens = question_tokens + gen_tokens
    for ref in ref_tokens:
        concatenated_tokens += ref
    
    # Compute the smoothed normalized tf-idf matrix for the concatenated tokens
    tfidf_matrix = compute_tfidf(concatenated_tokens)
    
    # Split the tf-idf matrix into submatrices
    question_tfidf = tfidf_matrix[:len(question_tokens)]
    gen_tfidf = tfidf_matrix[len(question_tokens):len(question_tokens)+len(gen_tokens)]
    ref_tfidf = tfidf_matrix[len(question_tokens)+len(gen_tokens):]
    
    # Compute the cosine similarities between the question and the generated answer and between the question and each reference answer
    question_gen_sim = cosine_similarity(question_tfidf, gen_tfidf)[0][0]
    question_ref_sim = cosine_similarity(question_tfidf, ref_tfidf)[0]
    
    # Compute the cosine similarities between the generated answer and each reference answer
    gen_ref_sim = cosine_similarity(gen_tfidf, ref_tfidf)[0]
    
    # Find the indices of the highest similarity scores
    max_question_gen_idx = question_gen_sim.argmax()
    max_question_ref_idx = question_ref_sim.argmax()
    max_gen_ref_idx = gen_ref_sim.argmax()
    
    # Print the results
    print("Question with the largest similarity to the ChatGPT-generated answer:")
    print("Question: ", ' '.join((map(str,question_tokens))))
    print("ChatGPT: ", ' '.join((map(str,gen_tokens))))
    print("Human: ", ' '.join((map(str,ref_tokens[max_gen_ref_idx]))))
    print()
    
    print("Question with the largest similarity to the human answer:")
    print("Question: ", ' '.join((map(str,question_tokens))))
    print("ChatGPT: ", ' '.join((map(str,gen_tokens))))
    print("Human: ", ' '.join((map(str,ref_tokens[max_question_ref_idx]))))
    print()
    
    print("Pair of ChatGPT-generated and human answers with the largest similarity:")
    print("ChatGPT: ", ' '.join((map(str,gen_tokens))))
    print("Human: ", ' '.join((map(str,ref_tokens[max_gen_ref_idx]))))
    print()
    
    # Compute the similarity scores as a pandas DataFrame
    result = pd.DataFrame({
        'question_ref_sim': question_ref_sim,
        'question_gen_sim': question_gen_sim,
        'gen_ref_sim': gen_ref_sim,
    })
    
    return result


# In[113]:


# Once case is tested here. 

question_tokens = tokenize(data["question"], lemmatized=False, remove_stopword=False, remove_punct = False)
gen_tokens = tokenize(data["chatgpt_answer"], lemmatized=False, remove_stopword=False, remove_punct = False)
ref_tokens = tokenize(data["human_answer"], lemmatized=False, remove_stopword=False, remove_punct = False)

result = assess_similarity(question_tokens, gen_tokens, ref_tokens)
result.head()

# You need to test other cases so that you can answer the analysis questions


# ## Q6 (Bonus): Further Analysis (Open question)
# 
# 
# - Can you find at least three significant differences between ChatGPT-generated and human answeres? Use data to support your answer.
# - Based on these differences, are you able to design a classifier to identify ChatGPT generated answers? Implement your ideas using traditional machine learning models, such as SVM, decision trees.
# 

# ## Test
# 
# Please move all your unit tests into the main block to make grading easy!

# In[115]:


if __name__ == "__main__":  
    
    # Test queston 1:
    question_tokens = tokenize(data["question"], lemmatized=False, remove_stopword=False, remove_punct = False)
    gen_tokens = tokenize(data["chatgpt_answer"], lemmatized=False, remove_stopword=False, remove_punct = False)
    ref_tokens = tokenize(data["human_answer"], lemmatized=False, remove_stopword=False, remove_punct = False)
    
    # Test question 2
    pos = pd.read_csv("/Users/yashitavajpayee/Documents/Semester II/BIA 660 - Web mining/positive-words.txt", header = None)
    neg = pd.read_csv("/Users/yashitavajpayee/Documents/Semester II/BIA 660 - Web mining/negative-words.txt", header = None)

    result = compute_sentiment(gen_tokens, 
                           ref_tokens, 
                           pos[0].values,
                           neg[0].values)
    print(result.head())
    
    # Test question 3
    result = bigram_precision_recall(gen_tokens, 
                                 ref_tokens)
    print(result.head())
    
    
    # Test question 4
    dtm = compute_tfidf(question_tokens)
    print(f"1.lemmatized=False, remove_stopword=False, remove_punct = False:\n     Shape: {dtm.shape}\n")
    
    # Test question 5
    result = assess_similarity(question_tokens, gen_tokens, ref_tokens)
    print(result.head())
    


# In[ ]:




