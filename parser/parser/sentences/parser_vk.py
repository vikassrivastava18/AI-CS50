import nltk
import sys

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | VP NP | NP VP P NP | S Conj S
NP -> N | Det N | NP Adv | P NP | NP P NP | Det AdjP NP
VP -> V | V NP | Adv VP | VP Adv    
AdjP -> Adj | Adj AdjP
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    s_lower = sentence.lower()

    token_list = nltk.word_tokenize(s_lower)

    # Remove unwanted characters
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    final_list = [word for word in token_list if word not in punc]
    return final_list


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    final = [subtree for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP')]
    # for t in tree:
    #     # tree.pretty_print()
    #
    #     for sub in t.subtrees():
    #         if sub.label() == 'NP':
    #             # get final 'NP'
    #             final_s_t =  get_final(sub)
    #
    #             # labels = []
    #             # for s in sub.subtrees()
    #             # if not any(label == 'NP' for label in labels):
    #             #     final.append(sub)
    return final

def get_final(sub_tree):
    for s in sub_tree.subtrees():
        if s.label() == 'NP':
            return get_final(s)


if __name__ == "__main__":
    main()
