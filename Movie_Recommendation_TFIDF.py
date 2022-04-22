import pandas as pd
import difflib
import qrcode
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, jsonify
from flask_restful import reqparse


app = Flask(__name__)

@app.route('/movierecom')

def movierecom():
    parser = reqparse.RequestParser()
    parser.add_argument('moviename', required=True, type=str)
    args = parser.parse_args()

    movies_data = pd.read_csv(r'C:\Users\dhild\OneDrive\Desktop\movies.csv')
    movies_data.head()
    
    movies_data.duplicated().sum()

    #selecting needed columns

    selected_features = set(['keywords','genres','tagline','cast','director','overview'])
    #print(selected_features)

    #replacing missing values with null string

    for feature in selected_features:
        movies_data[feature] = movies_data[feature].fillna('')

    #combining all the selected features

    combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['overview']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
    #print(combined_features)

    vectorizer = TfidfVectorizer()

    feature_vector = vectorizer.fit_transform(combined_features)

    similarity = cosine_similarity(feature_vector)

    movie_name = args['moviename']

    list_of_all_titles = movies_data['title'].tolist()

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    close_match = find_close_match[0]

    movie_index = movies_data[movies_data.title == close_match]['index'].values[0]
    
    similarity_score = list(enumerate(similarity[movie_index]))

    sorted_similar_movies = sorted(similarity_score,key = lambda x:x[1], reverse = True)

    recommended_movie = []

    i=1
    #print('Movies suggested for you: \n')
    for movie in sorted_similar_movies:
        index=movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]

        if(i<=4):
            recommended_movie.append(title_from_index)
            #print(i,'.',title_from_index)
            i=i+1
    
    result = {
        "rec_movie" : recommended_movie
    }

    return result



@app.route('/moviereview')
def moviereview():
    parser = reqparse.RequestParser()
    parser.add_argument('moviecomment', required=True, type=str)
    args = parser.parse_args()

    import nltk                                
    from nltk.corpus import movie_reviews     
    import matplotlib.pyplot as plt            
    import random
    import string
    from nltk.tokenize import word_tokenize
    from nltk import classify
    from nltk import NaiveBayesClassifier
    from random import shuffle 
    from nltk.corpus import movie_reviews 
    from nltk.corpus import stopwords
    from nltk import ngrams
    from nltk import FreqDist

    #nltk.download('movie_reviews')
    positive_review_file = movie_reviews.fileids('pos')[0] 
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((movie_reviews.words(fileid), category))
    shuffle(documents)
    all_words = [word.lower() for word in movie_reviews.words()]
    all_words_frequency = FreqDist(all_words)
    #nltk.download('stopwords')
    stopwords_english = stopwords.words('english')
    all_words_without_stopwords = [word for word in all_words if word not in stopwords_english]
    all_words_without_punctuation = [word for word in all_words if word not in string.punctuation]
    all_words_clean = []
    for word in all_words:
        if word not in stopwords_english and word not in string.punctuation:
            all_words_clean.append(word)
    all_words_frequency = FreqDist(all_words_clean)
    most_common_words = all_words_frequency.most_common(2000)
    word_features = [item[0] for item in most_common_words]
    def document_features(document):
        document_words = set(document)
        features = {}
        for word in word_features:
            features['contains(%s)' % word] = (word in document_words)
        return features
    movie_review_file = movie_reviews.fileids('neg')[0] 
    feature_set = [(document_features(doc), category) for (doc, category) in documents]
    pos_reviews = []
    for fileid in movie_reviews.fileids('pos'):
        words = movie_reviews.words(fileid)
        pos_reviews.append(words)
    neg_reviews = []
    for fileid in movie_reviews.fileids('neg'):
        words = movie_reviews.words(fileid)
        neg_reviews.append(words)
    stopwords_english = stopwords.words('english')
    def bag_of_words1(words):
        words_clean = []
        for word in words:
            word = word.lower()
            if word not in stopwords_english and word not in string.punctuation:
                words_clean.append(word)	
        words_dictionary = dict([word, True] for word in words_clean)	
        return words_dictionary
    pos_reviews_set = []
    for words in pos_reviews:
        pos_reviews_set.append((bag_of_words1(words), 'pos'))
    neg_reviews_set = []
    for words in neg_reviews:
        neg_reviews_set.append((bag_of_words1(words), 'neg'))
    #nltk.download('punkt')
    stopwords_english = stopwords.words('english')
    def clean_words(words, stopwords_english):
        words_clean = []
        for word in words:
            word = word.lower()
            if word not in stopwords_english and word not in string.punctuation:
                words_clean.append(word)	
        return words_clean 
    def bag_of_words(words):	
        words_dictionary = dict([word, True] for word in words)	
        return words_dictionary
    def bag_of_ngrams(words, n=2):
        words_ng = []
        for item in iter(ngrams(words, n)):
            words_ng.append(item)
        words_dictionary = dict([word, True] for word in words_ng)	
        return words_dictionary
    text = "It was a very good movie."
    words = word_tokenize(text.lower())
    words_clean = clean_words(words, stopwords_english)
    important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']
    stopwords_english_for_bigrams = set(stopwords_english) - set(important_words)
    words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)
    unigram_features = bag_of_words(words_clean)
    bigram_features = bag_of_ngrams(words_clean_for_bigrams)
    all_features = unigram_features.copy()
    all_features.update(bigram_features)
    def bag_of_all_words(words, n=2):
        words_clean = clean_words(words, stopwords_english)
        words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)
        unigram_features = bag_of_words(words_clean)
        bigram_features = bag_of_ngrams(words_clean_for_bigrams)
        all_features = unigram_features.copy()
        all_features.update(bigram_features)
        return all_features
    pos_reviews = []
    for fileid in movie_reviews.fileids('pos'):
        words = movie_reviews.words(fileid)
        pos_reviews.append(words)
    neg_reviews = []
    for fileid in movie_reviews.fileids('neg'):
        words = movie_reviews.words(fileid)
        neg_reviews.append(words)
    pos_reviews_set = []
    for words in pos_reviews:
        pos_reviews_set.append((bag_of_all_words(words), 'pos'))
    neg_reviews_set = []
    for words in neg_reviews:
        neg_reviews_set.append((bag_of_all_words(words), 'neg'))
    shuffle(pos_reviews_set)
    shuffle(neg_reviews_set)
    test_set = pos_reviews_set[:200] + neg_reviews_set[:200]
    train_set = pos_reviews_set[200:] + neg_reviews_set[200:]
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    custom_review = args['moviecomment']
    custom_review_tokens = word_tokenize(custom_review)
    custom_review_set = bag_of_all_words(custom_review_tokens)
    print (classifier.classify(custom_review_set))
    prob_result = classifier.prob_classify(custom_review_set)

    result = {
        "status":prob_result.max()
    }

    return jsonify(result)


@app.route('/qrgeneration/<string:bookingid>***<string:moviedate>***<string:movietime>')

def qrgeneration(bookingid,moviedate, movietime):
    img = qrcode.make(bookingid+","+moviedate+","+movietime)
    img.save("C:\\Users\\dhild\\OneDrive\\Desktop\\OMTB(7)\\OMTB(6)\\OMTB(4)\\OMTB(2)\moviess\\OMTB\\qrimages\\"+bookingid+".jpg")
    result = {
        "image path" : "qrimages/"+bookingid+".jpg"
        }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
