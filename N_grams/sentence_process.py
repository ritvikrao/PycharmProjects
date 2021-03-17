if __name__ == '__main__':
    f = open('hw2-my-test_orig.txt', 'r')
    sentences = f.readlines()
    sentences = [sentence.strip('\n') for sentence in sentences]
    sentences = [("<s> " + sentence.lower()) for sentence in sentences]
    sentences = [sentence + " </s>\n" for sentence in sentences]
    g = open('hw2-my-test.txt', 'w')
    g.writelines(sentences)
    f.close()
    g.close()