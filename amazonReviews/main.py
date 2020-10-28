import nltk
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia= SentimentIntensityAnalyzer()

df= pd.read_csv("amazonreviews.tsv", sep= '\t')
df.dropna(inplace=True)
spaces=[]
for index, label, review in df.itertuples():
	if type(review)==str:
		if review.isspace():
			spaces.append(review)

df.drop(spaces, inplace=True)
#check polarity score for first review
print(sia.polarity_scores(df.iloc[0]['review']))
df['scores']= df['review'].apply(lambda review: sia.polarity_scores(review))
df['compound']= df['scores'].apply(lambda d:d['compound'])
df['polarity']= df['compound'].apply(lambda compound: 'pos' if compound>=0 else 'neg')
print(df.head())
print(accuracy_score(df['label'], df['polarity']))
print(confusion_matrix(df['label'], df['polarity']))
print(classification_report(df['label'], df['polarity']))








