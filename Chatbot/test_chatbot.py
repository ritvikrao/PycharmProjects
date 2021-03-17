import unittest
import hw1_chatbot  # import your solution


# make sure that this file and your hw1_chatbot.py file
# are in the same directory, then run this file with the
# command `python test_chatbot.py`

# Ritvik Rao
# This is my unit test file for the chatbot

class ChatbotTest(unittest.TestCase):

    # you may also add set up functions if you'd like to, but are not
    # required to (and they aren't necessary for full credit)

    # Your tests here

    # Test forms of to be
    def test_to_be(self):
        examples = {"I was looking for you.": "You were looking for me.",
                    "We were looking for you.": "You were looking for me.",
                    "He was looking for them.": "He was looking for them.",
                    "I am looking for you.": "You are looking for me.",
                    "We are looking for you.": "You are looking for me.",
                    "He is looking for you.": "He is looking for me.",
                    "I have been thinking.": "You have been thinking."}

        chatter = hw1_chatbot.Chatbot()
        for ex_input in examples:
            response = chatter.respond(ex_input)
            self.assertEqual(examples[ex_input].lower(), response.lower(), msg="Expected string on left, but got " +
                                                                               "string on right.")

    # Important because there is a difference between these pronoun types
    # you is used as both the subject and object of the 2nd person
    def test_subject_vs_object(self):
        examples = {"I like you.": "You like me.",
                    "You like me.": "I like you.",
                    "I wanted to talk to you.": "You wanted to talk to me.",
                    "What did you want to say to me?": "What did I want to say to you?",
                    "They ran towards you.": "They ran towards me."}

        chatter = hw1_chatbot.Chatbot()
        for ex_input in examples:
            response = chatter.respond(ex_input)
            self.assertEqual(examples[ex_input].lower(), response.lower(), msg="Expected string on left, but got " +
                                                                               "string on right.")

    # Test to make sure third person things stay the same
    def test_third_person(self):
        examples = {"She likes pie.": "She likes pie.",
                    "He and I like to run.": "He and you like to run.",
                    "They want to go out tomorrow.": "They want to go out tomorrow.",
                    "They jumped and I jumped.": "They jumped and you jumped.",
                    "John went to the park.": "John went to the park.",
                    "It is good.": "It is good."}

        chatter = hw1_chatbot.Chatbot()
        for ex_input in examples:
            response = chatter.respond(ex_input)
            self.assertEqual(examples[ex_input].lower(), response.lower(), msg="Expected string on left, but got " +
                                                                               "string on right.")

    # Test for the sadness analysis
    def test_sadness_analysis(self):
        examples_no = {"He is feeling depressed.": "he is feeling depressed. I saw that you know someone who is "
                                                   "feeling: depressed. Do you want them to feel happy? Type yes or "
                                                   "no:",
                       "I am feeling bored.": "you are feeling bored. I saw that you are feeling: bored. Do you want "
                                              "to feel happy? Type yes or no:",
                       "They are angry.": "they are angry. I saw that you know some people who are feeling: angry. Do "
                                          "you want them to feel happy? Type yes or no:"}

        examples_yes = {"I am really really devastated.": "you are really really devastated. I saw that you are "
                                                          "feeling: devastated. Do you want to feel happy? Type yes "
                                                          "or no:",
                        "She is overwhelmed.": "she is overwhelmed. I saw that you know someone who is feeling: "
                                               "overwhelmed. Do you want them to feel happy? Type yes or no:",
                        "It is extremely disappointing to see you this way!": "it is extremely disappointing to see "
                                                                              "me this way! I saw that you know "
                                                                              "someone who is feeling: disappointing. "
                                                                              "Do you want them to feel happy? Type "
                                                                              "yes or no:"}

        chatter = hw1_chatbot.Chatbot()
        for ex_input in examples_no:
            response = chatter.special_respond(ex_input)
            self.assertEqual(examples_no[ex_input].lower(), response.lower(), msg="Expected string on left, but got " +
                                                                                  "string on right.")
            response = chatter.special_respond("no")
            self.assertEqual("no alright.", response.lower(), msg="Expected string on left, but got " +
                                                                  "string on right.")

        for ex_input_2 in examples_yes:
            response = chatter.special_respond(ex_input_2)
            self.assertEqual(examples_yes[ex_input_2].lower(), response.lower(),
                             msg="Expected string on left, but got " +
                                 "string on right.")
            response = chatter.special_respond("yes")
            self.assertEqual("yes here is a good source: https://www.boredbutton.com/", response.lower(),
                             msg="Expected string on left, but got " +
                                 "string on right.")

    # Test the expression evaluator
    def test_expression_evaluation(self):
        examples = {"5+5": "Looks like you want to solve a math problem. The answer to 5+5= 10",
                    "25* 5": "Looks like you want to solve a math problem. The answer to 25* 5= 125",
                    "4 - 3": "Looks like you want to solve a math problem. The answer to 4 - 3= 1",
                    "6 / 2 ": "Looks like you want to solve a math problem. The answer to 6 / 2= 3.0",
                    "5%2": "Looks like you want to solve a math problem. The answer to 5%2= 1",
                    "2 ** 3": "Looks like you want to solve a math problem. The answer to 2 ** 3= 8"}

        chatter = hw1_chatbot.Chatbot()
        for ex_input in examples:
            response = chatter.special_respond(ex_input)
            self.assertEqual(examples.get(ex_input).lower(), response.lower(),
                             msg="Expected string on left, but got string on right.")

    def test_provided(self):
        examples = {"My friend came to Northeastern today.": "Your friend came to Northeastern today.",
                    "I am happy.": "You are happy.",
                    "Why did I wait to do my homework?": "Why did you wait to do your homework?",
                    "My friend Devon had his halloween party. I went as a zebra.": "Your friend Devon had his " +
                                                                                   "halloween party. You went as a " +
                                                                                   "zebra.",
                    "We found a new tree. We named our tree \"Sam\".": "You found a new tree. You named your tree " +
                                                                       "\"Sam\".",
                    "I like talking to you. Your interface is friendly.": "You like talking to me. My interface is " +
                                                                          "friendly."}

        chatter = hw1_chatbot.Chatbot()
        for ex_input in examples:
            response = chatter.respond(ex_input)
            self.assertEqual(examples.get(ex_input).lower(), response.lower(),
                             msg="Expected string on left, but got string on right.")


if __name__ == "__main__":
    unittest.main()
