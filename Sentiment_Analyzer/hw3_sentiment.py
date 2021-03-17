# STEP 1: rename this file to hw3_sentiment.py

# feel free to include more imports as needed here
# these are the ones that we used for the base model
import numpy as np
import sys
import random
from collections import Counter

"""
Ritvik Rao

This is my implementation of the HW3 sentiment classifier
using Naive Bayes.
"""

"""
Cite your sources here:
Only source was lab 5 notebook
"""

"""
Gets folds of length k
No guarantee every single fold is length k:
k may not be divisible by length 
"""


def k_fold(all_examples, k):
    random.shuffle(all_examples)
    all_examples = [all_examples[i:i + k] for i in range(0, len(all_examples), k)]
    return all_examples


"""
Generates list of tuples
Each tuple: (ID, document_content, label)
Assumes the only tab characters in each line are
the ones separating the 3 parts in the tuple
"""


def generate_tuples_from_file(training_file_path):
    f = open(training_file_path, 'r')
    lines = f.readlines()
    lines = [line.strip('\n') for line in lines]
    f.close()
    docs = []
    for line in lines:
        new_tuple = tuple(line.split('\t'))
        if len(new_tuple) > 0:
            docs.append(new_tuple)
    return docs


"""
Helper method that classifies labels correctly
Returns array in following format:
[truepos, falsepos, trueneg, falseneg]
"""


def categorize(gold_labels, classified_labels):
    result = [0, 0, 0, 0]
    for i in range(len(gold_labels)):
        if gold_labels[i] == '0':
            if classified_labels[i] == '0':
                result[2] += 1
            else:
                result[1] += 1
        else:
            if classified_labels[i] == '0':
                result[3] += 1
            else:
                result[0] += 1
    return result


"""
Returns the precision as a float.
Precision=truepos/(truepos+falsepos)
"""


def precision(gold_labels, classified_labels):
    result = categorize(gold_labels, classified_labels)
    if result[0] + result[1] == 0:
        return 0
    return result[0] / (result[0] + result[1])


"""
Returns the recall as a float.
Precision=truepos/(truepos+falseneg)
"""


def recall(gold_labels, classified_labels):
    result = categorize(gold_labels, classified_labels)
    if result[0] + result[3] == 0:
        return 0
    return result[0] / (result[0] + result[3])


"""
Returns the F1 as a float.
F1=(2PR)/(P+R)
"""


def f1(gold_labels, classified_labels):
    prec = precision(gold_labels, classified_labels)
    rec = recall(gold_labels, classified_labels)
    if prec + rec == 0:
        return 0
    return (2 * prec * rec) / (prec + rec)

"""
SentimentAnalysis class
"""


class SentimentAnalysis:

    def __init__(self):
        # do whatever you need to do to set up your class here
        self.doc_count = [0, 0]  # [docs labelled '0', docs labelled '1']
        self.class_word_count = [0, 0]  # [words in '0', words in '1']
        self.wordcount = {}  # dictionary of: {word:[- count, + count]}

    """
    Train the model, returns nothing
    """

    def train(self, examples):
        for example in examples:
            content = example[1].split()
            # negative case
            if example[2] == '0':
                # add to the doc count
                self.doc_count[0] += 1
                for word in content:
                    # increment zero word count
                    self.class_word_count[0] += 1
                    if word in self.wordcount:
                        # add count to zero count for word
                        self.wordcount[word][0] += 1
                    else:
                        # initialize key
                        self.wordcount[word] = [1, 0]
            else:
                # add to the doc count
                self.doc_count[1] += 1
                for word in content:
                    # increment one word count
                    self.class_word_count[1] += 1
                    if word in self.wordcount:
                        # add count to one count for word
                        self.wordcount[word][1] += 1
                    else:
                        # initialize key
                        self.wordcount[word] = [0, 1]

    """
    Scores an example sentence based on the model
    calculates score of both possible labels and returns
    the results in dictionary format
    """

    def score(self, data):
        words = data.split()
        scores = {'0': 0, '1': 0}
        # calculate word score
        for i in range(2):
            log_sum = 0
            # add the prior
            log_sum += np.log(self.doc_count[i] / sum(self.doc_count))
            for word in words:
                # only go forward if the word is in the bag
                if word in self.wordcount:
                    # get count in class
                    class_count = self.wordcount[word][i]
                    # get number of words in class
                    class_size = self.class_word_count[i]
                    # get vocab size
                    vocab_size = len(self.wordcount)
                    # calculate score, add to sum
                    log_sum += np.log((class_count + 1) / (class_size + vocab_size))
            total_prob = np.e ** log_sum
            if i == 0:
                scores['0'] = total_prob
            else:
                scores['1'] = total_prob
        return scores

    """
    Finds the largest value for the given data
    either '0' or '1'
    """

    def classify(self, data):
        scores = self.score(data)
        if scores['1'] > scores['0']:
            return '1'
        else:
            return '0'

    """
    Initializes data as (word, True) tuples
    data = words separated by spaces
    static method
    """

    def featurize(self, data):
        new_tuples = []
        words = data.split()
        for word in words:
            new_tuples.append((word, True))
        return new_tuples

    def __str__(self):
        return "Naive Bayes - bag-of-words baseline"


class SentimentAnalysisImproved:

    def __init__(self):
        # get positive words
        f = open("positive_words.txt", "r")
        self.pos_words = f.readlines()
        self.pos_words = [line.strip('\n') for line in self.pos_words]
        f.close()
        # get negative words
        f = open("negative_words.txt", "r")
        self.neg_words = f.readlines()
        self.neg_words = [line.strip('\n') for line in self.neg_words]
        f.close()
        self.features = []  # list of list of features for each document
        self.weights = []  # list of weights for the entire dataset
        self.y = []  # current classes of each document

    """
    Add features with these methods
    separate methods to make ablation easier
    """

    def get_pos_words(self, words):
        pos_count = 0
        for word in words:
            if word in self.pos_words:
                pos_count += 1
        return pos_count

    def get_neg_words(self, words):
        neg_count = 0
        for word in words:
            if word in self.neg_words:
                neg_count += 1
        return neg_count

    def count_all_caps(self, words):
        all_caps = 0
        for word in words:
            if word == word.upper():
                all_caps += 1
        return all_caps

    def check_for_no(self, words):
        for word in words:
            word = word.strip(".?,\'\"")
            if word.lower() == "no":
                return 1
        return 0

    def check_for_yes(self, words):
        for word in words:
            word = word.strip(".?,\'\"")
            if word.lower() == "yes":
                return 1
        return 0

    def check_for_exclamation(self, words):
        for word in words:
            if "!" in word:
                return 1
        return 0

    def check_for_pronouns(self, words):
        pronoun_count = 0
        for word in words:
            if word.lower() == "i":
                pronoun_count += 1
            elif word.lower() == "me":
                pronoun_count += 1
            elif word.lower() == "mine":
                pronoun_count += 1
            elif word.lower() == "myself":
                pronoun_count += 1
            elif word.lower() == "you":
                pronoun_count += 1
            elif word.lower() == "yours":
                pronoun_count += 1
            elif word.lower() == "yourself":
                pronoun_count += 1
            elif word.lower() == "your":
                pronoun_count += 1
        return pronoun_count

    def get_word_count_log(self, words):
        return np.log(len(words))

    def sigmoid(self, z):
        return 1 / (1 + (np.e ** -z) )

    """
    Regularize
    """

    def regularize(self, vector):
        current_sum = 0
        for thing in vector:
            current_sum += thing ** 2
        return current_sum ** 0.5

    """
    Dot product
    """

    def dot_product(self, vector1, vector2):
        current_sum = 0
        for i in range(len(vector1)):
            current_sum += vector1[i] * vector2[i]
        return current_sum

    """
    Update the weights
    """

    def update_weights(self):
        converged_value = self.regularize(self.weights)  # previous converged value
        difference = 1
        learning_rate = 0.1
        max_epoch = 500
        epoch = 0
        m = len(self.y)  # number of examples
        while difference > 0.01:
            if epoch > max_epoch:
                break
            epoch += 1
            bias = self.weights[len(self.weights) - 1]  # last weight is bias
            # start by getting combined sigmoid
            current_gradients = []
            # gradient for each weight
            for i in range(len(self.weights)):
                this_gradient = 0
                for j in range(m):
                    featureset = self.features[j]
                    this_gradient += \
                        (self.sigmoid(self.dot_product(featureset, self.weights)) - self.y[j]) * featureset[i]
                this_gradient *= (1 / m)
                current_gradients.append(this_gradient)
            # once we get the gradients, update the weights
            for i in range(len(self.weights)):
                self.weights[i] = self.weights[i] - (learning_rate * current_gradients[i])
            # check if converged
            difference = abs(converged_value - self.regularize(self.weights))
            converged_value = self.regularize(self.weights)

    """
    Train the document
    Use helper methods to initialize features
    Then iterate over the dataset and update weights until convergence achieved
    """

    def train(self, examples):
        train_size = min(512, len(examples))
        for i in range(train_size):
            # initialize weight array
            feats = []
            words = examples[i][1].split()
            self.y.append(int(examples[i][2]))
            # append features
            # different methods to make ablation easier
            feats.append(self.get_pos_words(words))
            feats.append(self.get_neg_words(words))
            feats.append(self.count_all_caps(words))
            feats.append(self.check_for_yes(words))
            feats.append(self.check_for_no(words))
            feats.append(self.check_for_exclamation(words))
            feats.append(self.check_for_pronouns(words))
            feats.append(self.get_word_count_log(words))
            # bias feature
            feats.append(1)
            self.features.append(feats)
        # initialize weight arrays as all zeroes
        self.weights = [0 for i in range(len(self.features[0]))]
        # do the weight update
        self.update_weights()

    """
    Score given data with logistic regression weights
    Return P(+ | data)
    """

    def score(self, data):
        scores = {'0': 0, '1': 0}
        # build the feature array
        words = data.split()
        feats = []
        feats.append(self.get_pos_words(words))
        feats.append(self.get_neg_words(words))
        feats.append(self.count_all_caps(words))
        feats.append(self.check_for_yes(words))
        feats.append(self.check_for_no(words))
        feats.append(self.check_for_exclamation(words))
        feats.append(self.check_for_pronouns(words))
        feats.append(self.get_word_count_log(words))
        feats.append(1)
        plus_score = self.sigmoid(self.dot_product(feats, self.weights))
        scores['1'] = plus_score
        scores['0'] = 1 - plus_score
        return scores

    def classify(self, data):
        scores = self.score(data)
        if scores['1'] > scores['0']:
            return '1'
        else:
            return '0'

    def featurize(self, data):
        new_tuples = []
        words = data.split()
        for word in words:
            new_tuples.append((word, True))
        return new_tuples

    def __str__(self):
        return "Logistic Regression Classifier"

    def describe_experiments(self):
        s = """
    I chose to make this classifier a logistic regression classifier.
    It has the following features: 
    -The number of positive words, referenced by positive_words.txt from lecture 5 notebook
    -The number of negative words, references by negative_words.txt from lecture 5 notebook
    -The number of words in all caps
    -Does the document have "no"
    -Does the document have "yes"
    -Is there an exclamation point?
    -The number of 1st and 2nd person pronouns
    -The natural log of the total word count
    
    I then add these features and the bias feature (1) into an array for each document.
    I initialize the weights and bias for the dataset as 0.
    I then do batch training: the same thing as SGD, but on multiple examples.
    Batch training can get very expensive very quickly, so I limit the example count to 512.
    I calculated convergence by regularizing the weight array and seeing how much it
    changed each epoch.
    The effect of each feature was as follows, analyzed with feature ablation:
    
    -With all features on, precision, recall, and f1 are all 1.0
    -With positive words turned off, precision=0.7, recall=0.875, f1=0.77777
    -With negative words turned off, precision=0.636363, recall=0.875, f1=0.73684
    -With all caps turned off, precision, recall, and f1 are all 1.0
    -With yes check turned off, precision, recall, and f1 are all 1.0
    -With no check turned off, precision, recall, and f1 are all 1.0
    -With exclamation check turned off, precision, recall, and f1 are all 1.0
    -With pronouns turned off, precision, recall, and f1 are all 1.0
    -With word count turned off, precision=0.88888, recall=1.0, f1=0.941176
    -With no bias, precision, recall, and f1 are all 1.0
    
    As this is a sentiment classifier, the positive and negative word lists made the biggest
    difference. The word count also made a difference, and nothing else really did. I thought that the
    bias would be more important.
    
    """
        return s


def main():
    training = sys.argv[1]
    testing = sys.argv[2]

    sa = SentimentAnalysis()
    print(sa)

    # train analyzer
    sa.train(generate_tuples_from_file(training))
    # get tuples from testing data
    test_examples = generate_tuples_from_file(testing)
    # classify with model and get gold data
    gold_labels = []
    classified_labels = []
    for example in test_examples:
        gold_labels.append(example[2])
        classified_labels.append(sa.classify(example[1]))
    # report precision, recall, f1
    print("Precision: " + str(precision(gold_labels, classified_labels)))
    print("Recall: " + str(recall(gold_labels, classified_labels)))
    print("F1 Score: " + str(f1(gold_labels, classified_labels)))

    improved = SentimentAnalysisImproved()
    print(improved)
    # do the things that you need to with your improved class

    # train analyzer
    improved.train(generate_tuples_from_file(training))
    # get tuples from testing data
    test_examples = generate_tuples_from_file(testing)
    # classify with model and get gold data
    gold_labels = []
    classified_labels = []
    for example in test_examples:
        gold_labels.append(example[2])
        classified_labels.append(improved.classify(example[1]))
    # report precision, recall, f1
    print("Precision: " + str(precision(gold_labels, classified_labels)))
    print("Recall: " + str(recall(gold_labels, classified_labels)))
    print("F1 Score: " + str(f1(gold_labels, classified_labels)))
    print(improved.describe_experiments())

    # k-folds
    # combine data
    all_tuples = generate_tuples_from_file(training) + generate_tuples_from_file(testing)
    all_tuples = k_fold(all_tuples, 10)
    for i in range(10):
        print("Fold number " + str(i+1))
        dev_tuples = all_tuples[i]
        training_tuples = all_tuples[0:i] + all_tuples[i+1:]
        training_tuples = [new_tuple for train_list in training_tuples for new_tuple in train_list]
        analyzer = SentimentAnalysis()
        analyzer.train(training_tuples)
        gold_labels = []
        classified_labels = []
        for example in dev_tuples:
            gold_labels.append(example[2])
            classified_labels.append(analyzer.classify(example[1]))
        print("Precision: " + str(precision(gold_labels, classified_labels)))
        print("Recall: " + str(recall(gold_labels, classified_labels)))
        print("F1 Score: " + str(f1(gold_labels, classified_labels)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:", "python hw3_sentiment.py training-file.txt testing-file.txt")
        sys.exit(1)

    main()
