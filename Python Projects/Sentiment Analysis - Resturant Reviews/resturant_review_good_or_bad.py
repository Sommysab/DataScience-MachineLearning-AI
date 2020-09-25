import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('reviews.csv')


# VISUALS 
df['length'] = df['text'].apply(len)

df['length'].plot(bins=100, kind='hist')

sns.countplot(y='stars', data=df)

g = sns.FacetGrid(data=df, col='stars', col_wrap=3) 

g.map(plt.hist, 'length', bins=20, color='r') 


# New Record
df_1 = df[ df['stars'] == 1] 
df_5 = df[ df['stars'] == 5] 
df_1_5 = pd.concat([ df_1, df_5])


# CREATE, TEST, TRAIN & CLEAN 
import string as s

from nltk.corpus import stopwords as sw

def picker(message):
    Test_punc_removed = [char  for char in message if char not in s.puntuation ] 
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [ word for word in Test_punc_removed_join.split() if word.lower() not in sw.words('english')]
    return Test_punc_removed_join_clean


from sklearn.feature_extraction.text import CountVectorizer as CV
vectorizer = CV(analyzer = picker)
df_countvectorizer = vectorizer.fit_transform(df_1_5['text'])
label = df_1_5['stars'].values  


# NORM
X = df_countvectorizer
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


# Ex 1
testing_sample = ['amazing food! higly recommend it!']
testing_sample = vectorizer.transform(testing_sample)
test_predict = NB_c.predict(testing_sample)
print(test_predict)

# Ex 2
testing_sample = ['Shit food, made me sick']
testing_sample = vectorizer.transform(testing_sample)
test_predict = NB_c.predict(testing_sample)
print(test_predict) 

