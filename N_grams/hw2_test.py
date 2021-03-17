import unittest
from hw2_lm import LanguageModel

'''
This is the test file for hw2
'''


class TestMiniTraining(unittest.TestCase):
    def test_createunigrammodelnolaplace(self):
        unigram = LanguageModel(1, False)
        self.assertEqual(1, 1, msg="tests constructor for 1, False")

    def test_createunigrammodellaplace(self):
        unigram = LanguageModel(1, True)
        self.assertEqual(1, 1, msg="tests constructor for 1, True")

    def test_createbigrammodelnolaplace(self):
        bigram = LanguageModel(2, False)
        self.assertEqual(1, 1, msg="tests constructor for 2, False")

    def test_createbigrammodellaplace(self):
        bigram = LanguageModel(2, True)
        self.assertEqual(1, 1, msg="tests constructor for 2, True")

    def test_unigramlaplace(self):
        lm = LanguageModel(1, True)
        lm.train("training_files/iamsam.txt")
        # ((2 + 1) / (10 + 5))
        self.assertAlmostEqual(.2, lm.score("<s>"), msg="tests probability of <s>, trained on iamsam.txt")
        # ((2 + 1) / (10 + 5)) ** 2
        self.assertAlmostEqual(.04, lm.score("<s> </s>"), msg="tests probability of <s> </s>, trained on iamsam.txt")
        # ((2 + 1) / (10 + 5)) ** 3
        self.assertAlmostEqual(.008, lm.score("<s> i </s>"),
                               msg="tests probability of <s> i </s>, trained on iamsam.txt")

    def test_unigram(self):
        lm = LanguageModel(1, False)
        lm.train("training_files/iamsam2.txt")
        # ((4) / (20))
        self.assertAlmostEqual(.2, lm.score("<s>"), msg="tests probability of <s>, trained on iamsam2.txt")
        # (2 / 20)
        self.assertAlmostEqual(.1, lm.score("sam"), msg="tests probability of sam, trained on iamsam2.txt")
        # (4 / 20) * (2 / 20) * (4 / 20)
        self.assertAlmostEqual(.004, lm.score("<s> ham </s>"),
                               msg="tests probability of <s> ham </s>, trained on iamsam2.txt")

    def test_unigramunknowns(self):
        lm = LanguageModel(1, False)
        lm.train("training_files/iamsam.txt")
        # ((0) / (10))
        self.assertEqual(0, lm.score("flamingo"), msg="tests probability of flamingo, trained on iamsam.txt")

    def test_unigramunknownslaplace(self):
        lm = LanguageModel(1, True)
        lm.train("training_files/iamsam.txt")
        # ((0 + 1) / (10 + 5))
        self.assertAlmostEqual(0.0666667, lm.score("flamingo"), places=3,
                               msg="tests probability of flamingo, trained on iamsam.txt")

    def test_bigramunknowns(self):
        lm = LanguageModel(2, False)
        lm.train("training_files/iamsam.txt")
        # ((0) / (2))
        self.assertEqual(0, lm.score("<s> flamingo"), msg="tests probability of <s> flamingo, trained on iamsam.txt")

    def test_bigramunknownslaplace(self):
        lm = LanguageModel(2, True)
        lm.train("training_files/iamsam2.txt")
        # (0 + 1) / (4 + 6)
        self.assertAlmostEqual(0.1, lm.score("<s> flamingo"), places=3,
                               msg="tests probability of <s> flamingo, trained on iamsam2.txt")

    def test_bigram(self):
        lm = LanguageModel(2, False)
        lm.train("training_files/iamsam2.txt")
        # (2) / (4)
        self.assertAlmostEqual(.5, lm.score("<s> i"), msg="tests probability of <s> i, trained on iamsam2.txt")
        # (2 / 4) * (4 / 4) * (2 / 4)
        self.assertAlmostEqual(.25, lm.score("<s> i am </s>"),
                               msg="tests probability of <s> i am </s>, trained on iamsam2.txt")

    def test_bigramlaplace(self):
        lm = LanguageModel(2, True)
        lm.train("training_files/iamsam2.txt")
        # (2 + 1) / (4 + 6)
        self.assertAlmostEqual(.3, lm.score("<s> i"), msg="tests probability of <s> i, trained on iamsam2.txt")
        # ((2 + 1) / (4 + 6)) * ((4 + 1) / (4 + 6)) * ((2 + 1) / (4 + 6))
        self.assertAlmostEqual(.045, lm.score("<s> i am </s>"),
                               msg="tests probability of <s> i am </s>, trained on iamsam2.txt")

    def test_generatebigramconcludes(self):
        lm = LanguageModel(2, True)
        lm.train("training_files/iamsam2.txt")
        sents = lm.generate(2)
        self.assertEqual(2, len(sents), msg="tests that you generated 2 sentences and that generate concluded")
        print(sents)

    def test_generateunigramconcludes(self):
        lm = LanguageModel(1, True)
        lm.train("training_files/iamsam2.txt")
        sents = lm.generate(2)
        self.assertEqual(2, len(sents), msg="tests that you generated 2 sentences and that generate concluded")
        print(sents)

    def test_generatetrigramconcludes(self):
        lm = LanguageModel(4, True)
        lm.train("berp-training-four.txt")
        sents = lm.generate(50)
        for sent in sents:
            print(sent)
        self.assertEqual(50, len(sents), msg="tests that you generated 2 sentences and that generate concluded")

    def test_onlyunknownsgenerationandscoring(self):
        lm = LanguageModel(1, True)
        lm.train("training_files/unknowns.txt")

        # sentences should only contain unk tokens
        sents = lm.generate(5)
        for sent in sents:
            words = sent.split()
            if len(words) > 2:
                for word in words[1:-1]:
                    self.assertEqual("<UNK>", word.upper(),
                                     msg="tests that all middle words in generated sentences are <UNK>, unigrams")

        # probability of unk should be v high
        score = lm.score("porcupine")
        # (6 + 1) / (10 + 3)
        self.assertAlmostEqual(.5385, score, places=3,
                               msg="tests probability of porcupine, trained on unknowns.txt, unigrams")

        # and then for bigrams
        lm = LanguageModel(2, True)
        lm.train("training_files/unknowns.txt")

        # sentences should only contain unk tokens
        sents = lm.generate(5)
        for sent in sents:
            words = sent.split()
            if len(words) > 2:
                for word in words[1:-1]:
                    self.assertEqual("<UNK>", word.upper(),
                                     msg="tests that all middle words in generated sentences are <UNK>, bigrams")

        # probability of unk should be v high
        score = lm.score("porcupine wombat")
        # (6 + 1) / (10 + 3)
        self.assertAlmostEqual(.5555555, score, places=3,
                               msg="tests probability of porcupine wombat, trained on unknowns.txt, bigrams")


if __name__ == "__main__":
    unittest.main()
