# import any modules here
import re


# Ritvik Rao
# This is my implementation of the chatbot

class Chatbot:

    def __init__(self):

        # initialize negative_words.txt
        with open("negative_words.txt", 'r') as f:
            self.neg_words = f.readlines()
            self.neg_words = [line.strip('\n') for line in self.neg_words]

        # global for if user is feeling sad
        self.sad = False

        # initialize relations for easy replacements
        # initialize the subject relations

        # initialize the dependent relations
        self.dependent = {"my": "your",
                          "your": "my",
                          "our": "your"}

        # initialize the independent relations
        self.independent = {"mine": "yours",
                            "yours": "mine",
                            "ours": "yours"}

        # initialize the reflexives
        self.reflexives = {"myself": "yourself",
                           "yourself": "myself",
                           "ourselves": "yourselves",
                           "yourselves": "ourselves"}

    def respond(self, input):
        """
        Prompts the chatbot to respond to the given string.
        Parameters:
            input - string utterance from the user to respond to
        Return: string bot response
        """
        # start by turning everything to lowercase
        input = input.lower()
        response = []
        # split the tokens into sentences
        # this keeps the end-of-sentence punctuation
        # so that it can be added to the response
        sentences = re.split(r'([!?;.])', input)
        sentences = [x for x in sentences if x is not ""]
        # print(sentences)
        # main for loop
        for i in range(len(sentences)):
            # sentences are even numbered and
            # punctuation splits are odd numbered
            if i % 2 != 0:
                # rejoin the punctuation
                response[int(i / 2)] = "".join([response[int(i / 2)], sentences[i]])
            else:
                # build new string here
                newstring = sentences[i]
                # start by going through the easy cases with tokens
                # simple swap based on dictionaries
                tokens = re.split(r'(\W+)', sentences[i])
                for j in range(len(tokens)):
                    new_token = tokens[j]
                    if new_token in self.dependent:
                        new_token = self.dependent[new_token]
                    elif new_token in self.independent:
                        new_token = self.independent[new_token]
                    elif new_token in self.reflexives:
                        new_token = self.reflexives[new_token]
                    tokens[j] = new_token
                newstring = "".join(tokens)
                # then work with the more complex cases
                # split into clauses
                clauses = re.split(r'(\bthat\b)|(\band\b)|(\bbut\b)|(\bor\b)|(\bif\b)|(\bnot\b)|(\bbecause\b)',
                                   newstring)
                clauses = [x for x in clauses if x]
                # accumulate new sentence here
                newsentence = []
                for clause in clauses:
                    # split each clause into tokens
                    parts = re.split(r'( )', clause)
                    parts = [x for x in parts if x is not ""]
                    parts = [x for x in parts if x is not " "]
                    # check for contractions and split into parts
                    # n't and 's
                    for j in range(len(parts)):
                        if "n\'t" in parts[j] and parts[j] != "n\'t":
                            parts[j] = parts[j][0:len(parts[j]) - 3]
                            parts.insert(j + 1, "n\'t")
                        elif "\'s" in parts[j] and parts[j] != "\'s":
                            parts[j] = parts[j][0:len(parts[j]) - 2]
                            parts.insert(j + 1, "\'s")
                    for j in range(len(parts)):
                        # each of the following if statements represents
                        # a different "key word" as the first key word
                        # of a clause
                        # This gets most sentences right
                        # This works because I divided up
                        # each clause
                        # Key words: subj. and obj. pronouns
                        # also "to be" forms
                        # for example: if first meaningful word is "i"
                        if parts[j] == "i":
                            parts[j] = "you"  # always turn i to you
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":  # turn you to me in object case
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "was":
                                    parts[k] = "were"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "you":
                            parts[j] = "i"
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "are":
                                    parts[k] = "am"
                                elif parts[k] == "were":
                                    parts[k] = "was"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "we":
                            parts[j] = "you"
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "am":
                            parts[j] = "are"
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "are":
                            if parts[j + 1] == "you":
                                parts[j] = "am"
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "me":
                            parts[j] = "you"
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "us":
                            parts[j] = "you"
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "was":
                            if parts[j + 1] == "i":
                                parts[j] = "were"
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "were":
                            if parts[j + 1] == "you":  # were stays the same except if it is paired with you
                                parts[j] = "was"
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "i"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "to":
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                        elif parts[j] == "he" or parts[j] == "she" or parts[j] == "they" or parts[j] == "it":
                            for k in range(j + 1, len(parts)):
                                if parts[k] == "you":
                                    parts[k] = "me"
                                elif parts[k] == "me" or parts[k] == "us":
                                    parts[k] = "you"
                                elif parts[k] == "am":
                                    parts[k] = "are"
                                elif parts[k] == "i":
                                    parts[k] = "you"
                            break
                    # rejoin contractions
                    for j in range(len(parts) - 1, 0, -1):
                        if parts[j] == "n\'t":
                            parts[j - 1] = parts[j - 1] + "n\'t"
                            parts.remove("n\'t")
                        elif parts[j] == "\'s":
                            parts[j - 1] = parts[j - 1] + "\'s"
                            parts.remove("\'s")
                    newsentence.append(" ".join(parts))
                newstring = " ".join(newsentence)
                response.append(newstring)
        return " ".join(response)

    def special_respond(self, input):
        """
        Prompts the chatbot to respond to the given string.
        Parameters:
            input - string utterance from the user to respond to
        Return: string bot response
        """
        basic_response = self.respond(input)

        # Improvement 1: detect negative sentiment and cheer the user up
        # I am attempting to find out whether or not a user or someone
        # they know is sad. I then respond with the basic message, plus
        # a personal message from the bot added on with the correct
        # pronouns.
        # I am attempting to do this with a list of negative-sounding words.
        # I will look for "I feel", or "they feel", or "we feel", along with
        # a negative sounding word after.
        # Based on the results of this implementation, there is a wide array of
        # different possibilities for use, and the bot can even keep the conversation
        # going after one line. However, it is limited to a format of pronoun + verb
        # + some number of words (0 to n) + negative word. I felt that I would
        # get most of the intended sentences with this structure.

        if self.sad:
            if basic_response == "yes" or basic_response == "yes." or basic_response == "yes!":
                basic_response = basic_response + " Here is a good source: https://www.boredbutton.com/"
            else:
                basic_response = basic_response + " Alright."
            self.sad = False

        tokens = re.split(r'([ !?;.])', basic_response)
        tokens = [x for x in tokens if x is not " " or ""]
        for i in range(len(tokens) - 1):
            if (tokens[i] == "you") and ((tokens[i + 1] == "feel") or (tokens[i + 1] == "are")):
                for j in range(i, len(tokens)):
                    if tokens[j] in self.neg_words:
                        basic_response = basic_response + " I saw that you are feeling: " + tokens[j] + ". "
                        basic_response = basic_response + "Do you want to feel happy? Type yes or no:"
                        self.sad = True
            elif ((tokens[i] == "he") or (tokens[i] == "she") or (tokens[i] == "it")) \
                    and ((tokens[i + 1] == "feels") or (tokens[i + 1] == "is")):
                for j in range(i, len(tokens)):
                    if tokens[j] in self.neg_words:
                        basic_response = basic_response + " I saw that you know someone who is feeling: " + tokens[
                            j] + ". "
                        basic_response = basic_response + "Do you want them to feel happy? Type yes or no:"
                        self.sad = True
            elif (tokens[i] == "they") and ((tokens[i + 1] == "feel") or (tokens[i + 1] == "are")):
                for j in range(i, len(tokens)):
                    if tokens[j] in self.neg_words:
                        basic_response = basic_response + " I saw that you know some people who are feeling: " + tokens[
                            j] + ". "
                        basic_response = basic_response + "Do you want them to feel happy? Type yes or no:"
                        self.sad = True

        # Improvement 2: Detect operations
        # I am attempting to see if the user entered an operation and
        # then try to evaluate the expression
        # I will simply use the built-in eval() statement
        # and either return the result if eval returns correctly
        # or do nothing if it does not.
        # This will only work for expressions in the form
        # <number operand> <operator> <number operand>
        # but it does work

        try:
            result = eval(basic_response)
            basic_response = "Looks like you want to solve a math problem. The answer to " + basic_response + "= " + str(result)
        except:
            basic_response = basic_response

        return basic_response

    def greeting(self):
        """
        Prompts the chatbot to give an initial greeting.
        Return: string bot initial greeting
        """
        return "hello. i am boring bot."

    def __str__(self):
        return "Boring Bot"


def main():
    # Create a new chatbot
    cb = Chatbot()
    # the chatbot always begins by greeting the user
    begin = cb.greeting()
    print(cb, ":", begin)
    user_input = input("> ")

    # Any case of writing the word "exit" will cause the program to stop
    while user_input.lower() != "exit":
        bot_phrase = cb.respond(user_input)
        print(cb, ":", bot_phrase)
        user_input = input("> ")

    print("Goodbye!")


# This makes it so that the main function only runs when this file
# is directly run and not when it is imported as a module
if __name__ == "__main__":
    main()
