import pandas as pd
import numpy as np 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#opening the tsv file
df= pd.read_csv('moviereviews.tsv', sep= '\t')
#dropping the NAs
df.dropna(inplace=True)
#dropping the blanks in the reviews colomn 
df.dropna(inplace=True)
spaces=[]
for index, label, review in df.itertuples():
	if type(review)==str:
		if review.isspace():
			spaces.append(review)

df.drop(spaces, inplace=True)
#adding the sentiment polarity scores column 

sia=SentimentIntensityAnalyzer()

df['sentiment']=df['review'].apply(lambda review: sia.polarity_scores(review))
df['compound']=df['sentiment'].apply(lambda dico: dico['compound'])
#interpreting compount results ==> if >=0 pos else neg
df['polarity']=df['compound'].apply(lambda compound: 'pos' if compound >=0 else 'neg')

# test the performance using scikitlearn and the golds
print(accuracy_score(df['label'], df['polarity']))
print(confusion_matrix(df['label'], df['polarity']))
print(classification_report(df['label'], df['polarity']))
