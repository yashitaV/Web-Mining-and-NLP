#!/usr/bin/env python
# coding: utf-8

# # HW 5: Clustering and Topic Modeling

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. </div>

# In this assignment, you'll need to use the following dataset:
# - text_train.json: This file contains a list of documents. It's used for training models
# - text_test.json: This file contains a list of documents and their ground-truth labels. It's used for testing performance. This file is in the format shown below. Note, a document may have multiple labels.
# 
# 
# **Note: due to randomness, every time you run your clustering models, you may get different results. To ease the grading process, once you get satisfactory results, please save your notebook as a pdf file (Jupyter notebook menu File -> Print -> Save as pdf), and submit this pdf along with your .py code.**

# In[9]:


import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.cluster import KMeansClusterer
import numpy as np
import pandas as pd
from sklearn import metrics
from nltk.corpus import stopwords
from nltk.cluster import KMeansClusterer, cosine_distance
from nltk.cluster import KMeansClusterer
from sklearn.mixture import GaussianMixture
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[4]:


train_data = pd.read_csv("hw5_train.csv")
train_data.head()

test_data = pd.read_csv("hw5_test.csv")
test_data.head()


# ## Q1: K-Mean Clustering (5 points)
# 
# Define a function `cluster_kmean(train_data, test_data, num_clusters, min_df = 1, stopwords = None, metric = 'cosine')` as follows: 
# - Take two dataframes as inputs: `train_data` is the dataframe loaded from `hw5_train.csv`, and `test_data` is the dataframe loaded from `hw5_test.csv`
# - Use **KMeans** to cluster documents in `train_data` into 3 clusters by the distance metric specified. Tune the following parameters carefully:
#     - `min_df` and `stopword` options in generating TFIDF matrix. You may need to remove corpus-specific stopwords in addition to the standard stopwords.
#     - distance metric: `cosine` or `Euclidean` distance
#     - sufficient iterations with different initial centroids to make sure clustering converges
# - Test the clustering model performance using `test_data`: 
#     - Predict the cluster ID for each document in `test_data`.
#     - Apply `majority vote` rule to dynamically map each cluster to a ground-truth label in `test_data`. 
#         - Note a small percentage of documents have multiple labels. For these cases, you can randomly pick a label during the match
#         - Be sure `not to hardcode the mapping`, because a  cluster may corrspond to a different topic in each run. (hint: if you use pandas, look for `idxmax` function)
#     - Calculate `precision/recall/f-score` for each label. Your best F1 score on the test dataset should be around `80%`.
# - Assign a meaninful name to each cluster based on the `top keywords` in each cluster. You can print out the keywords and write the cluster names as markdown comments.
# - This function has no return. Print out confusion matrix, precision/recall/f-score. 
# 
# 
# **Analysis**:
# - Comparing the clustering with cosine distance and that with Euclidean distance, do you notice any difference? Which metric works better here?
# - How would the stopwords and min_df options affect your clustering results?

# In[5]:


# adding label col in test data
def get_label(row):
    if row['T1'] == 1:
        return 'T1'
    elif row['T2'] == 1:
        return 'T2'
    elif row['T3'] == 1:
        return 'T3'
    else:
        return None

test_data['label'] = test_data.apply(lambda row: get_label(row), axis=1)
test_data.head()


# In[74]:


def cluster_kmean(train_data, test_data, num_clusters, min_df=5, stopwords=None, metric='cosine'):
    
    tfidf_vect = TfidfVectorizer(stop_words="english", min_df=5) 
    dtm = tfidf_vect.fit_transform(train_data["text"])
    

    # initialize clustering model
    
    clusterer = KMeansClusterer(num_clusters, cosine_distance, repeats=20)

    # samples are assigned to cluster labels 
    clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)
    
    centroids = np.array(clusterer.means())

    # argsort sort the matrix in ascending order 
    # and return locations of features before sorting
    # [:,::-1] reverse the order
    sorted_centroids = centroids.argsort()[:, ::-1] 

    # The mapping between feature (word)
    # index and feature (word) can be obtained by
    # the vectorizer's function get_feature_names()
    voc_lookup = tfidf_vect.get_feature_names()

    for i in range(num_clusters):
        # get words with top 20 tf-idf weight in the centroid
        top_words = [voc_lookup[word_index] for word_index in sorted_centroids[i, :20]]
        print("Cluster %d:\n %s " % (i, "; ".join(top_words)))
              
    test_dtm = tfidf_vect.transform(test_data["text"])

    predicted = [clusterer.classify(v) for v in test_dtm.toarray()]
    confusion_df = pd.DataFrame(list(zip(test_data["label"].values, predicted)), columns=["label", "cluster"])

    print()
    
    # generate crosstab between clusters and true labels
    print(pd.crosstab(index=confusion_df.cluster, columns=confusion_df.label))      
    
    cluster_dict = {0:'T1', 1:"T2", 2:'T3'}

    # Map true label to cluster id
    predicted_target = [cluster_dict[i] for i in predicted]

    print()
    
    print(metrics.classification_report(test_data["label"], predicted_target))


# In[75]:


num_clusters = 3
cluster_kmean(train_data, test_data, num_clusters, min_df=5, metric='cosine')


# In[59]:


def cluster_kmean(train_data, test_data, num_clusters, min_df=1, stopwords=None, metric = 'euclidean'):
    tfidf_vect = TfidfVectorizer(stop_words="english", min_df=1) 
    dtm = tfidf_vect.fit_transform(train_data["text"])
    num_clusters = 3

    # initialize clustering model
    clusterer = KMeansClusterer(num_clusters, cosine_distance, repeats=20)

    # samples are assigned to cluster labels 
    # starting from 0
    clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)
    
    centroids = np.array(clusterer.means())

    # argsort sort the matrix in ascending order 
    # and return locations of features before sorting
    # [:,::-1] reverse the order
    sorted_centroids = centroids.argsort()[:, ::-1] 

    # The mapping between feature (word)
    # index and feature (word) can be obtained by
    # the vectorizer's function get_feature_names()
    voc_lookup = tfidf_vect.get_feature_names()

    for i in range(num_clusters):
        # get words with top 20 tf-idf weight in the centroid
        top_words = [voc_lookup[word_index] for word_index in sorted_centroids[i, :20]]
        print("Cluster %d:\n %s " % (i, "; ".join(top_words)))
              
    test_dtm = tfidf_vect.transform(test_data["text"])

    predicted = [clusterer.classify(v) for v in test_dtm.toarray()]
    confusion_df = pd.DataFrame(list(zip(test_data["label"].values, predicted)), columns=["label", "cluster"])

    print()
    # generate crosstab between clusters and true labels
    print(pd.crosstab(index=confusion_df.cluster, columns=confusion_df.label))      
    
    cluster_dict = {0:'T1', 1:"T2", 2:'T3'}

    # Map true label to cluster id
    predicted_target = [cluster_dict[i] for i in predicted]
    
    print()

    print(metrics.classification_report(test_data["label"], predicted_target))


# In[60]:


# Clustering by Euclidean distance
cluster_kmean(train_data, test_data, num_clusters, min_df=1, stopwords=None, metric='euclidean')


# ## Q2: GMM Clustering (5 points)
# 
# Define a function `cluster_gmm(train_data, test_data, num_clusters, min_df = 10, stopwords = stopwords)`  to redo Q1 using the Gaussian mixture model. 
# 
# **Requirements**:
# 
# - To save time, you can specify the covariance type as `diag`.
# - Be sure to run the clustering with different initiations to get stabel clustering results
# - Your F1 score on the test set should be around `70%` or higher.

# In[37]:


def cluster_gmm(train_data, test_data, num_clusters, min_df = 10, stopwords = stopwords):
    tfidf_vect = TfidfVectorizer(stop_words="english", min_df=10) 
    dtm = tfidf_vect.fit_transform(train_data["text"])
    
    lowest_bic = np.infty 
    best_gmm = None
    n_components_range = range(2, 5)  
    cv_types = ['spherical', 'tied', 'diag']
    
    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(dtm.toarray())
            bic = gmm.bic(dtm.toarray()) 
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
                 
    print('Best GMM is:', best_gmm)
    print()
    test_dtm = tfidf_vect.transform(test_data["text"])
    predicted = best_gmm.predict(test_dtm.toarray())
    
    new_df = pd.DataFrame(list(zip(test_data["label"].values, predicted)), columns=["actual_class", "cluster"])
    merged_df = pd.crosstab(index=new_df.cluster, columns=new_df.actual_class)
    print(merged_df)
    print()
    matrix = merged_df.idxmax(axis=1)
    final_predicted = [matrix[i] for i in predicted]
    classification_report = metrics.classification_report(test_data["label"], final_predicted)
    print(classification_report)


# In[38]:


cluster_gmm(train_data, test_data, num_clusters, min_df = 10, stopwords = stopwords)                                                                                                                            


# ## Q3: LDA Clustering (5 points)
# 
# **Q3.1.** Define a function `cluster_lda(train_data, test_data, num_clusters, min_df = 5, stopwords = stopwords)`  to redo Q1 using the LDA model. Note, for LDA, you need to use `CountVectorizer` instead of `TfidfVectorizer`. 
# 
# **Requirements**:
# - Your F1 score on the test set should be around `80%` or higher
# - Print out top-10 words in each topic
# - Return the topic mixture per document matrix for the test set(denoted as `doc_topics`) and the trained LDA model.

# **Q3.2**. Find similar documents
# 
# - Define a function `find_similar_doc(doc_id, doc_topics)` to find `top 3 documents` that are the most thematically similar to the document with `doc_id` using the `doc_topics`. (1 point)
# - Return the IDs of these similar documents.
# - Print the text of these documents to check if their thematic similarity.
# 
# 
# **Analysis**:
# 
# You already learned how to find similar documents by using TFIDF weights. Can you comment on the difference between the approach you just implemented with the one by TFID weights?

# In[43]:


def cluster_lda(train_data, test_data, num_clusters, min_df=5, stop_words_list=None):
    if stop_words_list is None:
        stop_words_list = nltk.corpus.stopwords.words('english')
    stop = list(stop_words_list) + ['said']
    count_vect = CountVectorizer(min_df=5, max_df=0.95, stop_words=stop)
    dtm_train = count_vect.fit_transform(train_data["text"])
    dtm_test = count_vect.transform(test_data["text"])
    tf_feature_names = count_vect.get_feature_names()

    lda_model = LatentDirichletAllocation(n_components=num_clusters, max_iter=30, verbose=1, evaluate_every=1, n_jobs=1, random_state=0).fit(dtm_train)
    doc_topics = lda_model.transform(dtm_test)
    topic_final = doc_topics.argmax(axis=1)
    data_df = pd.DataFrame({"actual_class": test_data["label"], "cluster": topic_final})
    data_df_converted = pd.crosstab(index=data_df.cluster, columns=data_df.actual_class)
    matrix = data_df_converted.idxmax(axis=1)
    final_predicted = [matrix[i] for i in topic_final]

    num_top_words = 10
    for topic_idx, topic in enumerate(lda_model.components_):
        print ("Topic %d:" % (topic_idx))
        words = [(tf_feature_names[i], topic[i]) for i in topic.argsort()[:-num_top_words-1:-1]]
        print(words)
        print("\n")

    print(data_df_converted)
    print(metrics.classification_report(test_data["label"], final_predicted))

    return doc_topics, lda_model


# In[44]:


num_clusters = 3
cluster_lda(train_data, test_data, num_clusters, min_df=5, stop_words_list=stopwords.words('english'))


# In[69]:


def find_similar(doc_id, doc_topics):
    # Get the topic mixture of the target document
    target_doc_topic_mix = doc_topics[doc_id]

    # Calculate the cosine similarity of the target document's topic mixture to all other documents' topic mixtures
    similarity_scores = []
    for i, doc_topic_mix in enumerate(doc_topics):
        similarity_score = np.dot(target_doc_topic_mix, doc_topic_mix) / (np.linalg.norm(target_doc_topic_mix) * np.linalg.norm(doc_topic_mix))
        similarity_scores.append((i, similarity_score))

    # Sort the similarity scores in descending order
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Return the top 3 most similar document IDs
    docs = [score[0] for score in similarity_scores[1:4]]

    return docs


# In[70]:


doc_topics[10:15]

doc_id = 11
idx = find_similar(doc_id, doc_topics)

print(test_data.text.iloc[doc_id])
print("Similar documents: \n")
for i in idx:
    print(i, test_data.iloc[i].text)


# ## Q4 (Bonus): Find the most significant topics in a document
# 
# A small portion of documents in our dataset have multiple topics. For instace, consider the following document which has topic T2 and T3. The LDA model returns two significant topics with probabilities 0.355 and 0.644. Can you describe a way to find out most significant topics in documents but ignore the insignificant ones? In this example, you should ignore the first topic but keep the last two.
# 
# - Implement your ideas
# - Test your ideas with the test set
# - Recalculate the precision/recall/f1 score for each label.
# 
# 

# In[135]:


(test_data.reset_index()).iloc[12:13]
doc_topics[12]


# In[ ]:


if __name__ == "__main__":  
    
    # Due to randomness, you won't get the exact result
    # as shown here, but your result should be close
    # if you tune the parameters carefully
    train = pd.read_csv("hw5_train.csv")
    train.head()

    test = pd.read_csv("hw5_test.csv")
    test.head()

    
    # Q1
    cluster_kmean(train_data, test_data, num_clusters, min_df=5, metric='cosine')
            
    # Q2
    
    cluster_gmm(train_data, test_data, num_clusters, min_df = 10, stopwords = stopwords)       
    


# In[ ]:




