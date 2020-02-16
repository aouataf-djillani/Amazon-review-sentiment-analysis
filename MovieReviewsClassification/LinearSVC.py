# classifiying movie reviews using the linear SVC model 



import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline 
from sklearn import metrics 

#Preparing the dataset 
df= pd.read_csv('moviereviews.tsv', sep='\t')

print(df.head())
print(df.isnull().sum())
df.dropna(inplace=True)

#removing empty strings 

blanks = [] 

for i,lb,rv in df.itertuples():  
    if type(rv)==str:            
        if rv.isspace():         
            blanks.append(i)     
        
df.drop(blanks, inplace=True)


#split: train test

X=df['review']
y=df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#pipeline(choose vectorizer, choose model)
my_model=Pipeline([('tfidf', TfidfVectorizer()),('classifier',LinearSVC())])
#the training 
my_model.fit(X_train,y_train)

#The testing 

predictions= my_model.predict(X_test)
print(metrics.confusion_matrix(y_test,predictions))
print(metrics.classification_report(y_test,predictions))
print(metrics.accuracy_score(y_test,predictions))

