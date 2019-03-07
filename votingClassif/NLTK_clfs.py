import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

all_words = []
documents = []

#allowed_word_types = ["J", "R", "V"]           ## H is adjective, r is adverb and v is verb
allowed_word_types = ["J"]

pos_paragraphs = short_pos.split('\n')
neg_paragraphs = short_neg.split('\n')

# Training paragraphs
pos_p = pos_paragraphs[:5000]
neg_p = neg_paragraphs[:5000]

for p in pos_p:
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in neg_p:
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("pickled_algos/documents.pickle", "wb")
pickle.dump(documents, save_documents)
save_documents.close()

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

save_word_features = open("pickled_algos/word_features5k.pickle", "wb")
pickle.dump(word_features, save_word_features)
save_word_features.close()


def find_features(documents):
    words = word_tokenize(documents)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


featuresets = [(find_features(rev), category) for (rev, category) in documents]
save_featuresets = open("pickled_algos/featuresets.pickle", "wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
print("Number of features processed: ", len(featuresets))

print("Classifiers:\nNaive Baies\nMultinomial Naive Baies\nBernoulli Naive Baies")
print("Logistic Regression\nLinear SVC\nSGD Classifier\nRandom Forest")

training_set = featuresets[:9000]
testing_set = featuresets[9000:]

# Pickle the test set to be used later for the voting classifier
save_set = open("test_set.pickle", "wb")
pickle.dump(testing_set, save_set)
save_set.close()

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes accuracy: ", (nltk.classify.accuracy(classifier,
                                                        testing_set)))
classifier.show_most_informative_features(15)

save_classifier = open("pickled_algos/naivebayes5k.pickle", "wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_Classifier accuracy", (nltk.classify.accuracy(MNB_classifier,
                                                         testing_set)))
save_classifier = open("pickled_algos/MNB_classifier5k.pickle", "wb")
pickle.dump(classifier, save_classifier)

save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier accuracy",
      (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)))

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle", "wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()


LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy: ",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)))

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle",
                       "wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()


LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy perncent: ",
      (nltk.classify.accuracy(LinearSVC_classifier, testing_set)))

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle", "wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()


SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDClassifier accuracy: ",
      nltk.classify.accuracy(SGDC_classifier, testing_set))

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle", "wb")
pickle.dump(SGDC_classifier, save_classifier)
save_classifier.close()


RF_classifier =  SklearnClassifier(RandomForestClassifier())
RF_classifier.train(training_set)
print("RF_classifier accuracy: ",
      nltk.classify.accuracy(RF_classifier, testing_set))

save_classifier = open("pickled_algos/RF_classifier5k.pickle", "wb")
pickle.dump(RF_classifier, save_classifier)
save_classifier.close()
