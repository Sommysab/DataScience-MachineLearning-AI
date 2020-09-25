import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 



spam_df = pd.read_csv('emails.csv')


# VISUAL
ham = spam_df[ spam_df['spam'] == 0]
spam = spam_df[ spam_df['spam'] == 1]

print('Spam Percentage =', (len(spam)/len(spam_df))*100, '%')
print('Ham Percentage =', (len(ham)/len(spam_df))*100, '%')

sns.countplot(spam_df['spam'], label = 'Count Spam vs. Ham')


# TEST, TRAIN & CLEAN
from sklearn.feature_extraction.text import CountVectorizer 

vectorizer = CountVectorizer()

spam_ham_countvectorizer = vectorizer.fit_transform(spam_df['text'])

label = spam_df['spam'].values 


from sklearn.naive_bayes import MultinomialNB 

NB_c = MultinomialNB() 
NB_c.fit(spam_ham_countvectorizer, label)

# Testing a Record

# EX 1
t_s = ['Free Money!! Grab and Grab for the last time!', 'Hi Obi, Please let me know if you need any further info.']
t_s_counvectorizer = vectorizer.transform(t_s)

prediction = NB_c.predict(t_s_counvectorizer)
print(prediction)


# EX 2
t_s = ['Hello, this is Sab. Am currently in town, lets met!', 'Make money, get Viagra!, be the superman.']
t_s_countvectorizer = vectorizer.transform(t_s)

prediction = NB_c.predict(t_s_countvectorizer)
print(prediction)


# NORM
X = spam_ham_countvectorizer
y = label
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

from sklearn.naive_bayes import MultinomialNB
NB_c = MultinomialNB() 
NB_c.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_predict_train = NB_c.predict(X_train) 
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


y_predict_test = NB_c.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))