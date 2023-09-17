import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import svm
import seaborn as sns
import re
import matplotlib.pyplot as plt
import missingno as ms
from xgboost import XGBClassifier
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

training_data = pd.read_csv('train.csv') 
testing_data = pd.read_csv('test.csv') 

#no of rows and columns in dataframe
print(training_data.shape, testing_data.shape)
#top 10 rows of data
print(training_data.head(10))
#complete information of dataframe
print(training_data.info())
#total number of positive and negative tweets
print(training_data['label'].value_counts()) 

#Exploratory Data Analysis
training_data['length'] = training_data['tweet'].apply(len)
fig1 = sns.barplot(x='label', y='length', data=training_data, palette='PRGn')
plt.title('Average Word Length vs label')
plot = fig1.get_figure()
plot.savefig('Barplot.png')

#bar graph to count positive negative label
fig2 = sns.countplot(x= 'label',data = training_data)
plt.title('Label Counts')
plot = fig2.get_figure()
plot.savefig('Count Plot.png')

#count of words disrtibution
plt.hist(training_data['length'], bins=100, edgecolor='black')
plt.xlabel('Word Count')
plt.ylabel('Count')
plt.title('Review Text Word Count Distribution')
plt.grid(True)
plt.ylim(0, 1600)
plt.xlim(0, 200)

def vectorization(table):
    #CountVectorizer will convert a collection of text documents to a matrix of token counts
    #Produces a sparse representation of the counts 
    #Initialize
    vector = CountVectorizer()
    #We fit and transform the vector created
    frequency_matrix = vector.fit_transform(table.tweet)
    #Sum all the frequencies for each word
    sum_frequencies = np.sum(frequency_matrix, axis=0)
    #Now we use squeeze to remove single-dimensional entries from the shape of an array that we got from applying np.asarray to
    #the sum of frequencies.
    frequency = np.squeeze(np.asarray(sum_frequencies))
    #Now we get into a dataframe all the frequencies and the words that they correspond to
    frequency_df = pd.DataFrame([frequency], columns=vector.get_feature_names_out()).transpose()
    return frequency_df


def graph(word_frequency, sent):
    labels = word_frequency[0][1:51].index
    title = "Word Frequency for %s" %sent
    #Plot the figures
    plt.figure(figsize=(10,5))
    plt.bar(np.arange(50), word_frequency[0][1:51], width = 0.8, color = sns.color_palette("bwr"), alpha=0.5, 
            edgecolor = "black", capsize=8, linewidth=1);
    plt.xticks(np.arange(50), labels, rotation=90, size=14);
    plt.xlabel("50 more frequent words", size=14);
    plt.ylabel("Frequency", size=14);
    #plt.title('Word Frequency for %s', size=18) %sent;
    plt.title(title, size=18)
    plt.grid(False);
    plt.gca().spines["top"].set_visible(False);
    plt.gca().spines["right"].set_visible(False);
    plt.savefig((sent+'.png'))


word_frequency = vectorization(training_data).sort_values(0, ascending = False)

#Graph with frequency words all, positive and negative tweets and get the frequency
graph(word_frequency, 'all')
#graph(word_frequency_pos, 'positive')
#graph(word_frequency_neg, 'negative')

word_frequency_pos = vectorization(training_data[training_data['label'] == 0]).sort_values(0, ascending = False)
word_frequency_neg = vectorization(training_data[training_data['label'] == 1]).sort_values(0, ascending = False)

graph(word_frequency_pos, 'positive')
graph(word_frequency_neg, 'negative')

def regression_graph(table):
    table = table[1:]
    #We set the style of seaborn
    sns.set_style("whitegrid")   
    #Initialize the figure
    plt.figure(figsize=(6,6))
    
    #we obtain the points from matplotlib scatter
    points = plt.scatter(table["Positive"], table["Negative"], c=table["Positive"], s=75, cmap="bwr")
    #graph the colorbar
    plt.colorbar(points)
    #we graph the regplot from seaborn
    sns.regplot(x="Positive", y="Negative",fit_reg=False, scatter=False, color=".1", data=table)
    plt.xlabel("Frequency for Positive Tweets", size=14)
    plt.ylabel("Frequency for Negative Tweets", size=14)
    plt.title("Word frequency in Positive vs. Negative Tweets", size=14)
    plt.grid(False)
    plt.savefig('Word-Freq_in_+ve_-ve.png')
    sns.despine()

table_regression = pd.concat([word_frequency_pos, word_frequency_neg], axis=1, sort=False)
table_regression.columns = ["Positive", "Negative"]
regression_graph(table_regression)

testing_data.head(10)  

#function to drop unwanted features
def drop_features(features,data):
    data.drop(features,inplace=True,axis=1)

def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())

training_data['processed_tweets'] = training_data['tweet'].apply(process_tweet)

training_data.head(15)

drop_features(['id','tweet'],training_data)
print(training_data.info())

x_train, x_test, y_train, y_test = train_test_split(training_data["processed_tweets"], training_data["label"], test_size = 0.2, random_state = 42)

count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)

x_train_counts = count_vect.fit_transform(x_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

print(x_train_counts.shape)
print(x_train_tfidf.shape)

print(x_train_counts)

x_test_counts = count_vect.transform(x_test)
x_test_tfidf = transformer.transform(x_test_counts)

print(x_test_counts.shape)
print(x_test_tfidf.shape)

model = RandomForestClassifier(n_estimators=500)
model.fit(x_train_tfidf,y_train)

predictions = model.predict(x_test_tfidf)

confusion_matrix(y_test,predictions)

rf_f1=f1_score(y_test,predictions)
print(rf_f1)

model_bow = XGBClassifier(random_state=22,learning_rate=0.9)
model_bow.fit(x_train_tfidf,y_train)

predict_xgb = model_bow.predict(x_test_tfidf)

confusion_matrix(y_test,predict_xgb)

xgb_f1=f1_score(y_test,predict_xgb)
print(xgb_f1)

lin_clf = svm.LinearSVC()
lin_clf.fit(x_train_tfidf,y_train)

predict_svm = lin_clf.predict(x_test_tfidf)

confusion_matrix(y_test,predict_svm)

svm_f1=f1_score(y_test,predict_svm)
svm_f1

results = {'RandomForest':rf_f1, 'XgBoost':xgb_f1,'SVM':svm_f1}  
df = pd.DataFrame(results, index =['f1Score']) 
print(df)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
tuned_parameters = {
    'vect__ngram_range': [(1, 1), (1, 2), (2, 2)],
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'clf__alpha': [1, 1e-1, 1e-2]
}

clf = GridSearchCV(text_clf, tuned_parameters, cv=10)
clf.fit(x_train, y_train)

print(classification_report(y_test, clf.predict(x_test), digits=4))