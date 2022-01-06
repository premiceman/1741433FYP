#Prem Vyas
import sys
import pandas
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score

df=pd.read_csv('fake-news/train.csv')
df
convert_val = {0: 'Real', 1: 'Fake'}
df['label'] = df['label'].replace(convert_val)
df.label.value_counts() #These values can show us how balanced the dataset is
print("-------------------Dataframe Sample-----------------------")
print()
print(df)

x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=7, shuffle=True)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.75)

vec_train=tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
vec_test=tfidf_vectorizer.transform(x_test.values.astype('U'))

pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(vec_train,y_train)

y_pred=pac.predict(vec_test)
score=accuracy_score(y_test, y_pred)
print(f'PAC Accuracy: {round(score*100,2)}%')

print(confusion_matrix(y_test,y_pred,labels=('Real','Fake')))

X=tfidf_vectorizer.transform(df['text'].values.astype('U'))

scores = cross_val_score(pac, X, df['label'].values, cv=5)
print(f'K Fold Accuracy: {round(scores.mean()*100,2)}%')

df_true=pd.read_csv('True.csv')
df_true['label']='Real'
#The line of code below redacts the publishers from the articles. We don't want this to skew the algorithms determination of whether the article is real or fake
df_true_rep=[df_true['text'][i].replace('WASHINGTON (Reuters) - ','').replace('LONDON (Reuters) - ','').replace('(Reuters) - ','') for i in range(len(df_true['text']))]
df_true['text']=df_true_rep
df_fake=pd.read_csv('Fake.csv')
df_fake['label']='Fake'
df_final=pd.concat([df_true,df_fake])
df_final=df_final.drop(['subject','date'], axis=1)
print(df_fake)

def findlabel(newtext):
    vec_newtest=tfidf_vectorizer.transform([newtext])
    y_pred1=pac.predict(vec_newtest)
    return y_pred1[0]

print()
print("-----------------------------------Article Prediction------------------------------------")

print(findlabel((df_true['text'][11])))


print()
print("-----------------------------------------------------------------------------------------")


realnewsacc = sum([1 if findlabel((df_true['text'][i]))=='Real' else 0 for i in range(len(df_true['text']))])/df_true['text'].size
print(f'Real News Dataset Classification Accuracy: {round(realnewsacc * 100,2)}%')

fakenewsacc = sum([1 if findlabel((df_fake['text'][i]))=='Fake' else 0 for i in range(len(df_fake['text']))])/df_fake['text'].size * 100
print(f'Fake News Dataset Classification Accuracy: {round(fakenewsacc,2)}%')
