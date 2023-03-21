import nltk
import sys
import os
import string

import numpy

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
    files = os.listdir(directory)
    output = {}
    sep = os.sep
    for file in files:
        path = os.path.join(directory, file)
        content = open(path, "r", encoding='utf-8')
        output[file] = content.read()

    return output


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    text = document.lower()
    text_list = nltk.word_tokenize(text)
    final_list = []
    for word in text_list:
        if not word in string.punctuation and not word in nltk.corpus.stopwords.words("english"):
            final_list.append(word)

    return final_list


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    result_dict = {}
    docs = documents.keys()

    for doc in docs:
        for word in documents[doc]:
            if not word in result_dict:
                count = 1
                for doc2 in docs:
                    if not doc == doc2:
                        if word in documents[doc2]:
                            count +=1

                result_dict[word] = numpy.log(len(docs)) - numpy.log(count)

    return result_dict


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf_scores = []
    for file in files.keys():
        score = 0
        for word in query:
            word_count = files[file].count(word)
            idf = idfs[word]
            score += word_count * idf
        tf_idf_scores.append({'file': file, 'score': score})
    tf_idf_scores = sorted(tf_idf_scores, key=lambda i: i['score'], reverse=True)
    output = []

    for i in range(n):
        output.append(tf_idf_scores[i]['file'])
    return output


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    matching_word_measure = []
    for sentence in sentences.keys():
        score = 0
        for word in query:
            if word in sentences[sentence]:
                score += idfs[word]
        matching_word_measure.append({'sentence': sentence, 'score': score})

    matching_word_measure = sorted(matching_word_measure, key=lambda i: i['score'], reverse=True)
    return matching_word_measure[:10]


if __name__ == "__main__":
    main()
