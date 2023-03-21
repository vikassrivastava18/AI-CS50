import nltk
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    file_content = dict()

    current_work_directory = os.getcwd()

    file_directory = os.path.join(os.sep,current_work_directory,directory)

    for f in os.listdir(file_directory):
        file_path = os.path.join(os.sep,file_directory,f)
        fo = open(file_path,"r", encoding='utf-8')
        file_content[f] = fo.read()
        

    return file_content


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    contents = []
    for word in nltk.word_tokenize(document.lower()) :
        if word in string.punctuation or word in nltk.corpus.stopwords.words("english"):
            continue   
        contents.append(word)

    return contents


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    word = set()

    #Add each word to a set with unique values
    for d in documents.keys():
        for w in documents[d] :
            word.add(w)

    for w in word :
        count = 0
        for d in documents.keys() :
            if w in documents[d] : 
                count += 1
        idfs[w] = math.log(len(documents)/count)
    
    return idfs

def term_frequency(word,list_of_words) :
    """
    Given a word, return the frequency of its occurance in given list of words
    """
    count = 0
    for w in list_of_words :
        if w == word :
            count += 1
    return count

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    d_tfidf = dict()
    l_files = []

    for f in files :
        tfidf = 0
        for w in query :
            if w in files[f] :
                tf = term_frequency(w,files[f])
                tfidf += tf * idfs[w]
        
        d_tfidf[f] = tfidf

    l = list(d_tfidf.values())
    l.sort(reverse = True)

    for s_tfidf in l :
        for name in list(d_tfidf.keys()) :
            if d_tfidf[name] == s_tfidf :
                if name not in l_files :
                    l_files.append(name)


    return l_files[:n]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    l_sentences = []
    d_idf = dict()

    for s in sentences :
        idf = 0
        for w in query :
            if w in sentences[s] :
                idf += idfs[w]
        
        d_idf[s] = idf

    list_idf = list(d_idf.values())
    list_unique_sorted_idf = list(set(list_idf))
    list_unique_sorted_idf.sort(reverse=True)

    for idf in list_unique_sorted_idf :

        # Find out how many sentences have the same idf value
        num_of_idf_repeat = sum(1 for i in list_idf if i == idf)

        if num_of_idf_repeat == 1:

            for s in list(d_idf.keys()):
                if d_idf[s] == idf:
                    if s not in l_sentences:
                        l_sentences.append(s)
        else :
            # Cases where IDF value is same for multiple sentences, prioritize the sentences that have a higher query term density.

            #Cumulate all the sentences with the same IDF
            list_of_sentence_same_idf = list(s for s in d_idf if d_idf[s] == idf)

            d_term_density = dict()
            # Store IDF for these sentences in a dictionary
            for s in list_of_sentence_same_idf:
                d_term_density[s] = sum(1 for w in query if w in sentences[s])/len(sentences[s])
            # Append the sentences in the return list, preferring sentences with higher term density

            l_term_density = list(d_term_density.values())
            l_term_density.sort(reverse=True)

            for td in l_term_density:
                for s in list(d_term_density.keys()):
                    if d_term_density[s] == td:
                        if s not in l_sentences:
                            l_sentences.append(s)

    return l_sentences[:n]


if __name__ == "__main__":
    main()
