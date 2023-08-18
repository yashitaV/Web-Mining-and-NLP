#!/usr/bin/env python
# coding: utf-8

# # <Center> HW 4: Classification </center>

# <div class="alert alert-block alert-warning">Each assignment needs to be completed independently. Never ever copy others' work (even with minor modification, e.g. changing variable names). Anti-Plagiarism software will be used to check all submissions. </div>

# In this assignment, we build a classifier to identify ChatGPT-generated or human answers. The dataset is taken from https://huggingface.co/datasets/Hello-SimpleAI/HC3. Two files have been prepared for you for training and testing
# - hw4_train.csv: dataset for training
# - hw4_test.csv: dataset for testing. 
#     
# The label column indicates if the answer is ChatGPT-generated (label 1) or human answer
# 

# In[1]:


import pandas as pd

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[2]:


data_train = pd.read_csv("/Users/yashitavajpayee/Documents/Semester II/BIA 660 - Web mining/HW 4/hw4_train.csv")
data_train.head()


# In[3]:


data_test = pd.read_csv("/Users/yashitavajpayee/Documents/Semester II/BIA 660 - Web mining/HW 4/hw4_test.csv")
data_test.head()


# ## Q1 Classification
# 
# - Define a function `create_model(train_docs, train_y, test_docs, test_y, model_type='svm', stop_words='english', min_df = 1, print_result = True)`, where
# 
#     - `train_docs`: is a list of documents for training
#     - `train_y`: is the ground-truth labels of the training documents
#     - `test_docs`: is a list of documents for test
#     - `test_y`: is the ground-truth labels of the test documents
#     - `model_type`: two options: `nb` (Multinomial Naive Bayes) or `svm` (Linear SVM)
#     - `stop_words`: indicate whether stop words should be removed. The default value is 'english', i.e. remove English stopwords.
#     - `min_df`: only word with document frequency above this threshold can be included. The default is 1. 
#     - `print_result`: controls whether to show classification report or plots. The default is True.
# 
# 
# - This function does the following:
#     - Fit a `TfidfVectorizer` using `train_docs` with options `stop_words, min_df` as specified in the function inputs. Extract features from `train_docs` using the fitted `TfidfVectorizer`.
#     - Train `linear SVM` or `Multinomial Naive Bayes` model as specified by `model_type` using the extracted features and `train_y`. 
#     - Tranform `test_docs` by the fitted `TfidfVectorizer` (hint: use function `transform` not `fit_transform`).
#     - Predict the labels for `test_docs`. If `print_result` is True, print the classification report.
#     - Calculate the AUC score and PRC scores for class 1 on the test dataset. If `print_result` is True, plot the ROC and PRC curves. **Hint**: 
#         - `sklearn.svm.LinearSVM` does not provide `predict_proba` function. 
#         - Instead, you can use its `decision_function` (see <a href = "https://stackoverflow.com/questions/59227176/how-to-plot-roc-and-calculate-auc-for-binary-classifier-with-no-probabilities-s">some reference code</a>) 
#         - Another option is to use `sklearn.svm.SVC` with `kernel='linear' and probability=False` (see <a href = "https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html"> reference</a>. This option can be very slow.)
#     - Return the trained model, the fitted `TfidfVectorizer`, and the AUC and PRC scores.
# 

# - Test your function with the `answer` column as the input text in following cases:
#     - model_type='svm', stop_words = 'english', min_df = 1
#     - model_type='nb', stop_words = 'english', min_df = 1

# In[20]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import scipy
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import cosine_similarity
from warnings import filterwarnings


# In[5]:


def create_model(train_docs, train_y, test_docs, test_y, model_type='svm', stop_words='english', min_df=1, print_result=True):
    
    # Fit TfidfVectorizer using train_docs
    tfidf_vect = TfidfVectorizer(stop_words=stop_words, min_df=min_df)
    tfidf_vect.fit(train_docs)
    
    # Extract features from train_docs using the fitted TfidfVectorizer
    train_X = tfidf_vect.transform(train_docs)
    
    # Train linear SVM or Multinomial Naive Bayes model
    if model_type == 'svm':
        model = LinearSVC()
    elif model_type == 'nb':
        model = MultinomialNB()
    else:
        raise ValueError("Invalid model type. Model type should be 'svm' or 'nb'.")
    model.fit(train_X, train_y)

    # Tranform test_docs by the fitted TfidfVectorizer
    test_X = tfidf_vect.transform(test_docs)

    # Predict the labels for test_docs
    y_pred = model.predict(test_X)
    
    # Calculate the AUC score and PRC scores for class 1 on the test dataset
    if model_type == 'svm':
        y_score = model.decision_function(test_X)
    elif model_type == 'nb':
        y_score = model.predict_proba(test_X)[:, 1]
    else:
        raise ValueError("Invalid model type. Model type should be 'svm' or 'nb'.")
    
    fpr, tpr, _ = roc_curve(test_y, y_score, pos_label=1)
    precision, recall, _ = precision_recall_curve(test_y, y_score, pos_label=1)
    auc_score = (auc(fpr, tpr))*100
    prc_score = (auc(recall, precision))*100
    
    # Print the classification report and plot ROC and PRC curves
    if print_result:
        print("Classification report:\n", classification_report(test_y, y_pred))
        
        # Plot ROC curve
        plt.figure(figsize=(6,4))
        lw = 2
        plt.plot(fpr, tpr, color='darkorange')
        plt.plot([0, 1], [0, 1], color='blue', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.show()
        
        # Plot PRC curve
        plt.figure(figsize=(6,4))
        lw = 2
        plt.plot(recall, precision, color='blue')
        plt.xlabel('Recall')
        plt.ylabel('Precision')        
        plt.title('PRC')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.show()
        
        
        print('AUC score:', auc_score)
        print('PRC score:', prc_score)
    return model, tfidf_vect, auc_score, prc_score


# In[6]:


train_docs = data_train['answer'].tolist()
train_y = data_train['label'].tolist()
test_docs = data_test['answer'].tolist()
test_y = data_test['label'].tolist()


# In[7]:


# Test SVM model

create_model(train_docs, train_y, test_docs, test_y,               model_type='svm', stop_words=None, min_df = 1, print_result = True)


# In[8]:


# Test Naive Bayes model

create_model(train_docs, train_y, test_docs, test_y,               model_type='nb', stop_words=None, min_df = 1, print_result = True)


# ## Q2: Search for best parameters 
# 
# From Task 1, you may find there are many possible ways to configure parameters. Next, let's use grid search to find the optimal parameters
# 
# - Define a function `search_para(docs, y, model_type = 'svm')` where 
#     - `docs` are training documents
#     - `y` is the ground-truth labels
#     - `model_type`: either SVM or Naive Bayes classifier
# - This function does the following:
#     - Create a pipleline which integrates `TfidfVectorizer` and the classifier 
#     - Define the parameter ranges as follow:
#         - `stop_words': [None, 'english']`
#         - `min_df: [1,2,3, 5]`
#     - Set the scoring metric to "f1_macro"
#     - Use `GridSearchCV` with `5-fold cross validation` to find the best parameter values based on the training dataset. 
#     - Print the best parameter values
#     
# - For each SVM or Naive Bayes model, call the function `create_model` defined in Task 1 `with the best parameter values`. 
# 
# 
# - `Analysis`: Please briefly answer the following:
#     - Compare with the model in Task 1, how is the performance improved on the test dataset?
#     - Why do you think the new parameter values help sentiment classification?

# In[9]:


def search_para(docs, y, model_type):
    
    if model_type == 'svm':
        classifier = LinearSVC(class_weight='balanced')
    else:
        classifier = MultinomialNB()
    
    pipe = Pipeline([('tfidf', TfidfVectorizer()), ('classifier', classifier)])
    
    param_grid = {
        'tfidf__stop_words': [None, 'english'],
        'tfidf__min_df': [1, 2, 3, 5]
    }
    
    f1_scorer = make_scorer(f1_score, average='macro')
    search = GridSearchCV(pipe, param_grid, cv=5, scoring=f1_scorer)
    search.fit(docs, y)
    
    
    print(search.best_params_)
    print("Best f1 score:", search.best_score_)
    


# In[10]:


# Best parameter for SVM

docs = train_docs
y = train_y
search_para(docs, y, model_type='svm')


# In[11]:


# Retrain SVM with the best parameters

best_params = {'tfidf__min_df': 2, 'tfidf__stop_words': None}
svm_pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words=None, min_df=5)), ('svm', LinearSVC())])

# train the model on the entire training set
svm_pipeline.fit(train_docs, train_y)

# evaluate the model on the test set
create_model(train_docs, train_y, test_docs, test_y, model_type='svm', stop_words=None, min_df=2, print_result=True)


# In[12]:


# Best parameter for SVM

docs = train_docs
y = train_y
search_para(docs, y, model_type='nb')


# In[13]:


# set the best parameters obtained from grid search
best_params = {'tfidf__min_df': 5, 'tfidf__stop_words': None}

# create the pipeline
nb_pipeline = Pipeline([('tfidf', TfidfVectorizer()), ('nb', MultinomialNB())])

# set the pipeline parameters to the best values obtained from grid search
nb_pipeline.set_params(**best_params)

# train the model on the entire training set
nb_pipeline.fit(train_docs, train_y)

# evaluate the model on the test set
create_model(train_docs, train_y, test_docs, test_y,               model_type='nb', stop_words=None, min_df = 5, print_result = True)


# ## Q3: Improved Classifier
# 
# So far we only considered the TFIDF weights as features. Can you considered other features, e.g. the differences you noticed in HW3, and incorporate these features into your classifier? Your target is to `improve the F1 macro score of your SVM or Naive Bayes by at least 2%`.

# In[29]:


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score, classification_report, roc_curve


def create_advanced_model(train_df, train_y, test_df, test_y,                           model_type='svm', stop_words=None, min_df=1, print_result=True):
    
    # Define additional features
    def get_word_count(docs):
        return np.array([len(doc.split()) for doc in docs]).reshape(-1, 1)

    def get_char_count(docs):
        return np.array([len(doc) for doc in docs]).reshape(-1, 1)

    def get_capital_count(docs):
        return np.array([sum(1 for c in doc if c.isupper()) for doc in docs]).reshape(-1, 1)

    def get_punctuation_count(docs):
        return np.array([sum(doc.count(p) for p in '.,;:!') for doc in docs]).reshape(-1, 1)

    # Combine the additional features with the TF-IDF features
    feature_union = FeatureUnion([
        ('tfidf', TfidfVectorizer(stop_words=stop_words, min_df=min_df)),
        ('word_count', FunctionTransformer(get_word_count, validate=False)),
        ('char_count', FunctionTransformer(get_char_count, validate=False)),
        ('capital_count', FunctionTransformer(get_capital_count, validate=False)),
        ('punctuation_count', FunctionTransformer(get_punctuation_count, validate=False)),
    ])

    # Define the pipeline with the feature union and the classifier
    if model_type == 'svm':
        clf = LinearSVC(class_weight='balanced')
    elif model_type == 'nb':
        clf = MultinomialNB()
    else:
        print('Invalid model type')
        return None, None, None, None
    
    pipeline = Pipeline([
        ('features', feature_union),
        ('clf', clf)
    ])
    
    # Fit the pipeline on the training data
    pipeline.fit(train_df, train_y)

    # Predict on the test data
    y_pred = pipeline.predict(test_df)
    
    # Compute evaluation metrics
    f1 = f1_score(test_y, y_pred, average='macro')
    auc = roc_auc_score(test_y, y_pred, average='macro')
    prc = average_precision_score(test_y, y_pred, average='macro')

    if print_result:
        print('F1 score:', f1)
        print('AUC score:', auc)
        print('PRC score:', prc)
        print('\nClassification Report:')
        print(classification_report(test_y, y_pred))

        # Plot ROC curve
        fpr, tpr, thresholds = roc_curve(test_y, y_pred, pos_label=1)
        plt.plot(fpr, tpr, label='Auc curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Auc')
        plt.legend(loc="lower right")
        plt.show()

    return pipeline, feature_union, auc, prc


# In[30]:


svm_model, svm_feature_union, svm_auc, svm_prc = create_advanced_model(train_docs, train_y, test_docs, test_y,                                                                        model_type='svm', stop_words=None, min_df=5,                                                                        print_result=True)


# ## Q4 (Bonus): Model Interpretation
# 
# Take the best-performing model you achieve in Task 2, can you identify the most important words that can differentiate human answers from ChatGPT generated answers?
# 
# 
# For both SVM and Naive Bayes models, describe your idea, implement your idea, and show the top 20 most descrimiating words.

# In[ ]:


if __name__ == "__main__":  
     
    # add test code

