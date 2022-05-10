'''
utils_text.py : Collection of Text Utility Functions
'''

from gensim.models import Phrases
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import string


def remove_phrases(df_col: pd.DataFrame, phrases: list) -> pd.DataFrame:
    ''' Remove phrases from dataframe column 
    :param: df_col: pandas dataframe column
    :param: phrases: list of phrases to remove 
    :return: pandas dataframe column with phrases removed '''
    for phrase in phrases:
        df_col = df_col.str.replace(phrase, '', regex=False)

    return df_col


def remove_punctuations(text: str) -> str:
    ''' Remove punctuations from text 
    :param: text: string
    :return: string with punctuations removed '''
    # "’", "‘", "—", "…", "“", "”", and "–" are added in addition for removal.
    pattern = r"[{}{}]".format(string.punctuation, '’‘—…“”–·•')
    return text.translate(str.maketrans('', '', pattern))


def remove_stopwords(text: str, extra_stopwords: list = None) -> str:
    ''' Remove stop words that do not carry useful information. 
    :param: text: string 
    :param: extra_stopwords: list of extra stop words 
    :return: string with stop words removed '''
    stop_words = set(stopwords.words('english'))

    if extra_stopwords != None:
        stop_words.update(extra_stopwords)

    word_tokens = word_tokenize(text)
    
    filtered = []
    for word in word_tokens:
        if (word not in stop_words):
            filtered.append(word)

    return ' '.join(filtered)


def stem_words(text: str)-> str:
    ''' Remove affixes from words using PorterStemmer. 
    :param: text: string of words
    :return: string of stemmed words '''
    ps = PorterStemmer()
    text = [ps.stem(word) for word in word_tokenize(text)]

    return ' '.join(text)


def lemmatize_words(text: str) -> str:
    ''' Convert words to the base form
    while ensuring the converted words are part of the language. 
    :param: text: string of words 
    :return: string of lemmatized words '''
    wnl = WordNetLemmatizer()

    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    results = []
    for word, tag in pos_tag(word_tokenize(text)):
        # print("word: {}, tag: {}, tag_dict[tag[0]]: {}".format(word, tag, tag_dict[tag[0]]))
        if tag[0] in tag_dict:
            lemma = wnl.lemmatize(word, tag_dict[tag[0]])
            results.append(lemma)
        else:
            results.append(word)
    
    return ' '.join(results)


def vectorize_words(raw_documents):
    ''' Creates a matrix of word vectors. 
    :param: raw_documents: iterable of strings
    :return: vectorizer and word vectors '''
    cv = CountVectorizer(binary=True)
    cv.fit(raw_documents)
    matrix = cv.transform(raw_documents)

    return cv, matrix


def get_ngrams_df(df, text_col,  degree_ngrams):
    ''' Get a dataframe containing n-grams and frequencies. 
    :param: df: pandas dataframe 
    :param: textcol: name of text column in df
    :param: degree_ngram: degree of n-grams (e.g. bigrams, trigrams, 4-grams) 
    :return: dataframe containing n-grams and frequencies. '''
    sentences = [sentence.split() for sentence in df[text_col]]

    words = []
    for i in range(0, len(sentences)):
        words += sentences[i]

    # Create a df containing n-grams and frequencies.
    df = pd.Series(ngrams(words, degree_ngrams)).value_counts()
    df = df.to_frame().reset_index()
    df = df.rename(columns={'index': 'N-Gram', 0: 'Frequency'})
    df = df.astype({'N-Gram': str})

    # Remove brackets, quotes, and commas from n-grams.
    pattern = ',|\'|\\(|\\)'
    df['N-Gram'] = df['N-Gram'].str.replace(pattern, '', regex=True)

    return df


def process_texts_for_gensim(text_col):
    ''' Prepare texts for gensim models (e.g., Word2Vec, LDA). 
    :param: text_col: pandas dataframe text column
    :return: list of lists of tokenized texts for gensim models '''
    lines = '\n'.join(text_col).split('\n')

    # Create tokenized texts
    texts = [line.lower().split(' ') for line in lines]

    # Add bigrams and trigrams to texts
    bigrams = Phrases(texts, min_count=20, threshold=2, delimiter='_')
    trigrams = Phrases(bigrams[texts], min_count=20, threshold=2, delimiter='_') 
    for i in range(len(texts)):
        for token in trigrams[texts[i]]:
            # If token is bigram/trigram, add it to texts
            if '_' in token:
                texts[i].append(token)

    return texts


if __name__ == "__main__":
    pass
