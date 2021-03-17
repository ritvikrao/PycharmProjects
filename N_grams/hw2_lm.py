# imports go here
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np

"""
Ritvik Rao

This is my main implementation of the
language model
"""


# Feel free to implement helper functions

class LanguageModel:
    # constants to define pseudo-word tokens
    # access via self.UNK, for instance
    UNK = "<UNK>"
    SENT_BEGIN = "<s>"
    SENT_END = "</s>"

    def __init__(self, n_gram, is_laplace_smoothing):
        """Initializes an untrained LanguageModel
        Parameters:
          n_gram (int): the n-gram order of the language model to create
          is_laplace_smoothing (bool): whether or not to use Laplace smoothing
        """
        self.n_gram = n_gram
        self.is_laplace_smoothing = is_laplace_smoothing
        self.vocabulary = {"<s>": 0, "</s>": 0}
        self.ngram_list = {}


    """
    Makes the ngrams based on the length of the ngram.
    Parameters: tokens for the tokens in a sentence
    Returns: none
    """
    def make_ngrams(self, tokens):
        # if the ngram is longer than the sentence,
        # add the whole sentence to the model
        if self.n_gram > len(tokens):
            new_ngram = tuple(tokens)
            if new_ngram not in self.ngram_list:
                self.ngram_list[new_ngram] = 1
            else:
                self.ngram_list[new_ngram] += 1
        else:
            for i in range(len(tokens) - (self.n_gram - 1)):
                new_ngram = tuple(tokens[i:i+self.n_gram])
                if new_ngram not in self.ngram_list:
                    self.ngram_list[new_ngram] = 1
                else:
                    self.ngram_list[new_ngram] += 1


    def train(self, training_file_path):
        """Trains the language model on the given data. Assumes that the given data
        has tokens that are white-space separated, has one sentence per line, and
        that the sentences begin with <s> and end with </s>
        Parameters:
          training_file_path (str): the location of the training data to read

        Returns:
        None
        """
        # start by reading the file
        f = open(training_file_path, 'r')
        sentences = f.readlines()
        sentences = [sentence.strip('\n') for sentence in sentences]
        # then add words/counts to the vocabulary
        # count the words
        for sentence in sentences:
            tokens = sentence.split()
            # count up all the words
            for i in range(len(tokens)):
                if tokens[i] not in self.vocabulary:
                    self.vocabulary[tokens[i]] = 1
                else:
                    self.vocabulary[tokens[i]] += 1
        # second pass to replace frequency 1 counts
        # with UNK
        # then make the n-grams
        for sentence in sentences:
            tokens = sentence.split()
            for i in range(len(tokens)):
                if tokens[i] is not (self.SENT_BEGIN or self.SENT_END):
                    if self.vocabulary[tokens[i]] == 1:
                        self.vocabulary.pop(tokens[i])
                        tokens[i] = self.UNK
                        # if UNK is not in the vocab, add it
                        if self.UNK not in self.vocabulary:
                            self.vocabulary[self.UNK] = 1
                        else:
                            self.vocabulary[self.UNK] += 1
            # make the n-grams
            self.make_ngrams(tokens)


    """
    Calculates the probability that a single n-gram appears
    based on the model.
    :param this_ngram the ngram input as a tuple
    :return the probability as a float
    """
    def ngram_probability(self, this_ngram):
        # Case 1: unigram
        if self.n_gram == 1:
            ngram_count = 0
            if this_ngram in self.ngram_list:
                ngram_count = self.ngram_list[this_ngram]
            word_count = sum(self.vocabulary.values())
            # Case 1.1: Laplace smoothing
            if self.is_laplace_smoothing:
                return (ngram_count + 1) / (len(self.vocabulary) + word_count)
            # Case 1.2: no smoothing
            else:
                return ngram_count / word_count
        # Case 1: multigram
        else:
            # get the ngram prefix
            ngram_prefix = this_ngram[:len(this_ngram) - 1]
            # count the number of times the prefix occurs in the ngram list
            prefix_count = 0
            for ngram in self.ngram_list:
                if ngram_prefix == ngram[:len(this_ngram) - 1]:
                    prefix_count += self.ngram_list[ngram]
            # count the number of times the n-gram occurs in the list
            ngram_count = 0
            if this_ngram in self.ngram_list:
                ngram_count = self.ngram_list[this_ngram]
            # Case 2.1: Laplace smoothing
            if self.is_laplace_smoothing:
                return (ngram_count + 1) / (prefix_count + len(self.vocabulary))
            # Case 2.2: no smoothing
            else:
                return ngram_count / prefix_count

    def score(self, sentence):
        """Calculates the probability score for a given string representing a single sentence.
        Parameters:
          sentence (str): a sentence with tokens separated by whitespace to calculate the score of

        Returns:
          float: the probability value of the given string for this model
        """
        # method: probability score is equal to
        # probability of the sentence's n-grams
        # multiplied together
        tokens = sentence.split()
        # preprocess the sentence for unknowns
        for i in range(len(tokens)):
            if tokens[i] not in self.vocabulary:
                tokens[i] = self.UNK
        # if the sentence is too short,
        # no chance it's in the model
        if (len(tokens)) < self.n_gram:
            return 0.0
        else:
            total_probability = 0
            for i in range(len(tokens) - (self.n_gram - 1)):
                # get the n-gram
                this_ngram = tuple(tokens[i:i + self.n_gram])
                probability = self.ngram_probability(this_ngram)
                # instantly return 0 if a probability is 0
                # avoids error when taking the log of probability
                if probability == 0.0:
                    return 0
                additional = math.log(probability)
                total_probability = total_probability + additional
            return math.e ** total_probability

    def generate_sentence(self):
        """Generates a single sentence from a trained language model using the Shannon technique.

        Returns:
          str: the generated sentence
        """
        # Unigram strategy
        if self.n_gram == 1:
            # start sentence with s
            newsentence = ['<s>']
            # Start generation with word count
            # without count for '<s>'
            word_count = sum(self.vocabulary.values())
            word_count = word_count - self.vocabulary['<s>']
            # add words
            while newsentence[len(newsentence) - 1] != '</s>':
                # generate a random number
                target = random.randrange(word_count)
                current_count = 0
                for word in self.vocabulary:
                    if word != "<s>":
                        current_count += self.vocabulary[word]
                        if current_count > target:
                            newsentence = newsentence + [word]
                            break
            return " ".join(newsentence)
        # Multigram strategy
        else:
            # start sentence with <s> n-1 times
            newsentence = []
            for i in range(self.n_gram - 1):
                newsentence.append('<s>')
            # start loop
            while newsentence[len(newsentence) - 1] != '</s>':
                # begin by getting a count of every time the prefix shows up
                prefix_count = 0
                for ngram in self.ngram_list:
                    if tuple(newsentence[len(newsentence) - (self.n_gram - 1):len(newsentence)]) == ngram[:len(ngram) - 1]:
                        prefix_count += self.ngram_list[ngram]
                    # get a random number based off the prefix count
                target = random.randrange(prefix_count)
                current_count = 0
                for an_ngram in self.ngram_list:
                    if tuple(newsentence[len(newsentence) - (self.n_gram - 1):len(newsentence)]) == an_ngram[
                                                                                                    :len(an_ngram) - 1]:
                        current_count += self.ngram_list[an_ngram]
                        if current_count > target:
                            # append last ngram word
                            newsentence.append(an_ngram[len(an_ngram) - 1])
                            break
            # before return, append extra </s>
            for i in range(self.n_gram - 2):
                newsentence.append(self.SENT_END)
            return " ".join(newsentence)

    def generate(self, n):
        """Generates n sentences from a trained language model using the Shannon technique.
        Parameters:
          n (int): the number of sentences to generate

        Returns:
          list: a list containing strings, one per generated sentence
        """
        return[self.generate_sentence() for i in range(n)]

    """
    Finds the perplexity of the word sequence provided.
    Multiply probabilities of all n-grams appearing.
    Then take the Nth root of the reciprocal of the product
    where N is the number of N-grams.
    :param test_sequence the sequence of tokens as a string
    :return the perplexity
    
    def perplexity(self, test_sequence):
        # Each n-gram has a probability
    """


def main():
    # this main method will print the histograms
    # for all of the test data
    # 4 models: unigram or bigram, laplace or no laplace
    model_uni_no_laplace = LanguageModel(1, False)
    model_uni_laplace = LanguageModel(1, True)
    model_bi_no_laplace = LanguageModel(2, False)
    model_bi_laplace = LanguageModel(2, True)
    # Train all models
    model_uni_no_laplace.train(sys.argv[1])
    model_uni_laplace.train(sys.argv[1])
    model_bi_no_laplace.train(sys.argv[1])
    model_bi_laplace.train(sys.argv[1])
    # get scores for hw2-test.txt
    f = open(sys.argv[2], 'r')
    sentences = f.readlines()
    f.close()
    sentences = [sentence.strip('\n') for sentence in sentences]
    test1_model_uni_no_laplace = [model_uni_no_laplace.score(sentence) for sentence in sentences]
    test1_model_uni_laplace = [model_uni_laplace.score(sentence) for sentence in sentences]
    test1_model_bi_no_laplace = [model_bi_no_laplace.score(sentence) for sentence in sentences]
    test1_model_bi_laplace = [model_bi_laplace.score(sentence) for sentence in sentences]
    # get scores for hw2-my-test.txt
    g = open(sys.argv[3], 'r')
    sentences2 = g.readlines()
    g.close()
    sentences2 = [sentence.strip('\n') for sentence in sentences2]
    test2_model_uni_no_laplace = [model_uni_no_laplace.score(sentence) for sentence in sentences2]
    test2_model_uni_laplace = [model_uni_laplace.score(sentence) for sentence in sentences2]
    test2_model_bi_no_laplace = [model_bi_no_laplace.score(sentence) for sentence in sentences2]
    test2_model_bi_laplace = [model_bi_laplace.score(sentence) for sentence in sentences2]
    # Print random sentences
    print("Generated sentences for unigram: ")
    autogenerated = model_uni_no_laplace.generate(50)
    for sentence in autogenerated:
        print(sentence)
    print("Generated sentences for bigram: ")
    autogenerated = model_bi_no_laplace.generate(50)
    for sentence in autogenerated:
        print(sentence)
    # generate histograms unigram
    logbins = np.logspace(np.log10(10 ** -100), np.log10(10 ** 0), 100)
    plt.xscale('log')
    plt.hist([test1_model_uni_no_laplace, test1_model_uni_laplace, test2_model_uni_no_laplace, test2_model_uni_laplace],
             bins=logbins, label=["hw2-test-no-laplace",
                                                                                   "hw2-test-laplace",
                                                                                   "hw2-my-test-no-laplace",
                                                                                   "hw2-my-test-laplace"],
             stacked=True)
    plt.xlabel('Probability score for the sentence')
    plt.ylabel('Number of sentences')
    plt.title(r'Unigram Model Frequencies')
    plt.legend()
    plt.savefig("hw2-unigram-histogram.pdf", bbox_inches="tight")
    plt.clf()
    # generate histograms bigram
    plt.xscale('log')
    plt.hist([test1_model_bi_no_laplace, test1_model_bi_laplace, test2_model_bi_no_laplace, test2_model_bi_laplace],
             bins=logbins, label=["hw2-test-no-laplace",
                                  "hw2-test-laplace",
                                  "hw2-my-test-no-laplace",
                                  "hw2-my-test-laplace"],
             stacked=True)
    plt.xlabel('Probability score for the sentence')
    plt.ylabel('Number of sentences')
    plt.title(r'Bigram Model Frequencies')
    plt.legend()
    plt.savefig("hw2-bigram-histogram.pdf", bbox_inches="tight")


if __name__ == '__main__':

    # make sure that they've passed the correct number of command line arguments
    if len(sys.argv) != 4:
        print("Usage:", "python hw2_lm.py training_file.txt textingfile1.txt textingfile2.txt")
        sys.exit(1)

    main()

