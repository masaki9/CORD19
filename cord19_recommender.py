'''
cord19_recommender.py : Recommend COVID-19 transmission related articles based on the query
'''

import numpy as np
import pandas as pd
import utils_text as utils_text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('data/cord19_transmission_processed.csv', header=0, sep=',')

def clean_query(query:str) -> str:
    ''' Clean the query. 
    :param: query: string
    :return: cleaned query '''
    query = query.lower()
    query = utils_text.lemmatize_words(query)
    query = utils_text.remove_punctuations(query)
    query = utils_text.remove_stopwords(query)

    return query

def recommend(query:str, top_k:int=5) -> list:
    ''' Recommend k most relevant articles. 
    :param: query: string
    :param: top_k: number of top recommended articles 
    :return: list of top recommended articles '''
    vectorizer = CountVectorizer()
    papers_matrix = vectorizer.fit_transform(df['abstract_processed'])

    cleaned_query = clean_query(query)

    query_matrix = vectorizer.transform([cleaned_query]) # Pass list of 1 string as transform accepts iterable
    # print("Query Matrix Shape: {}".format(query_matrix.shape))

    cosine_sim = cosine_similarity(query_matrix, papers_matrix)
    # print("Cosine Similaries Matrix Shape: {}".format(cosine_sim.shape))

    cosine_sim = cosine_sim.flatten() # flatten 2d to 1d as num of rows = 1 (query)
    # print("Similarities from Highest to Lowest: {}".format(np.sort(cosine_sim)[::-1]))

    indices = np.argsort(cosine_sim)[::-1] # get indices for similarties and sort them descendingly
    # indices = np.argsort(-cosine_sim) # get indices for similarties and sort them descendingly
    top_k_indices = indices[0:top_k]

    print("\n##################")
    print("Recommended Papers")
    print("##################\n")

    recommended_papers = []
    for i in top_k_indices:
        print("Paper ID: {}\nTitle: {}\nAbstract: {}\n".format(df['paper_id'].at[i], df['title'].at[i], df['abstract'].at[i]))
        recommended_papers.append((df['paper_id'].at[i], df['title'].at[i], df['abstract'].at[i]))

    return recommended_papers


recommend('COVID-19 household transmission', top_k=4)
recommend('COVID-19 Omicron transmissibility', top_k=3)
recommend('COVID-19 vertical transmission', top_k=1)
recommend('COVID-19 fomite transmission', top_k=1)
