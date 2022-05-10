'''
cord19_etl_non_trans.py : Extract, transform, and load non transmission related COVID-19 articles
'''

from elasticsearch import Elasticsearch, helpers
import os
import pandas as pd
import utils_text as utils_text

es = Elasticsearch('http://localhost:9200', api_key=os.environ['ES_API_KEY'])

# Get documents whose title contains one of the known COVID-19 names and does not match regex ".*transmi.*|.*infect.*" 
# and whose abstract does not match regex ".*transmi.*|.*infect.*". Set range of publish time to limit num of docs.
body={
    'query': {
        "bool":{
            "must": [
                {
                    "bool": {
                        "minimum_should_match": 1,
                        "should" : [
                            {"match" : { "title" : "COVID-19" }},
                            {"match" : { "title" : "coronavirus disease 2019" }},
                            {"match" : { "title" : "SARS-CoV-2" }},
                            {"match" : { "title" : "severe acute respiratory syndrome coronavirus 2" }},
                            {"match" : { "title" : "2019 novel coronavirus" }}
                        ]
                    }
                },
                {
                    "bool": {
                        "must_not" : {
                            "regexp": {
                                "title" : ".*transmi.*|.*infect.*"
                            }
                        }
                    }
                },
                {
                    "bool": {
                        "must_not" : {
                            "regexp": {
                                "abstract" : ".*transmi.*|.*infect.*"
                            }
                        }
                    }
                },           
                {
                            'range': {
                                'publish_time': {
                                'gte': '2022-01-01',
                                'lte': '2022-03-31',
                                'format': 'yyyy-MM-dd'
                                }
                            }
                }
            ]
        }
    }
}
results = helpers.scan(es, query=body, index='cord19', scroll='5m', preserve_order=True)

# Only get data in _source and store it in dataframe
df = pd.DataFrame.from_dict([document['_source'] for document in results])
df = df[["paper_id","title", "abstract", "text_body"]]
print(df.shape)

# Remove documents with no abstracts
df = df.mask(df['abstract'] == '')
df = df.dropna(axis=0, how="any")
print(df.shape)

df_transmi = pd.read_csv('data/cord19_transmission_processed.csv', header=0, sep=',')

# Match the size of df to that of the other dataset
df = df[0:df_transmi.shape[0]]
print(df.shape)

# Convert text to lowercase for text processing.
df['abstract_processed'] = df['abstract'].str.lower()

# Remove URLs
df['abstract_processed'] = df['abstract_processed'].str.replace('http\S+|www\S+', '', regex=True)

meaningless = ['\n', 'made available', 'international license', 'publisher\'s note', 'used to guide clinical practice', 'open access',
               'springer nature remains neutral with regard to jurisdictional claims in published maps and institutional affiliations',
               'creative common', 'reproduction in any medium', 'provided the original work is properly cited', 'cohort study',
               'case study', 'the following', 'display the preprint', 'word count', 'publicly available', 'all rights reserved',
               'no reuse allowed without permission', 'copyright holder', 'peer review']

df['abstract_processed'] = utils_text.remove_phrases(df['abstract_processed'], meaningless)

df['abstract_processed'] = df['abstract_processed'].str.replace('severe acute respiratory syndrome coronavirus 2', 'sars-cov-2', regex=False)
df['abstract_processed'] = df['abstract_processed'].str.replace('severe acute respiratory syndrome', 'sars', regex=False)
df['abstract_processed'] = df['abstract_processed'].str.replace('coronavirus 2', 'cov-2', regex=False)
df['abstract_processed'] = df['abstract_processed'].str.replace('coronavirus disease 2019', 'covid-19', regex=False)
df['abstract_processed'] = df['abstract_processed'].str.replace('middle east respiratory syndrome', 'mers', regex=False)
df['abstract_processed'] = df['abstract_processed'].str.replace('middle eastern respiratory syndrome', 'mers', regex=False)

df['abstract_processed'] = df['abstract_processed'].str.replace('sars-cov-2 sars-cov-2', 'sars-cov-2', regex=False)
df['abstract_processed'] = df['abstract_processed'].str.replace('sars sars', 'sars', regex=False)
df['abstract_processed'] = df['abstract_processed'].str.replace('cov-2 cov-2', 'cov-2', regex=False)
df['abstract_processed'] = df['abstract_processed'].str.replace('covid-19 covid-19', 'covid-19', regex=False)
df['abstract_processed'] = df['abstract_processed'].str.replace('mers mers', 'mers', regex=False)

df['abstract_processed'] = df['abstract_processed'].apply(utils_text.remove_punctuations)
df['abstract_processed'] = df['abstract_processed'].apply(utils_text.lemmatize_words)

# Remove single characters and digits
df['abstract_processed'] = df['abstract_processed'].str.replace(r'\b[a-z]\b', '', regex=True)
df['abstract_processed'] = df['abstract_processed'].str.replace(r'\b\d+\b', '', regex=True)

words = ['preprint', 'copyright', 'author', 'elsevier', 'biorxiv', 'pmc', 'medrxiv', 'publisher', 'pubmed', 
         'ccbyncnd', 'ccbyncd', 'ccbync', 'ccbynd', 'byncnd', 'ccby', 'cc', 'nc', 'nd', 'ncd',  'doi', 'figure', 'fig', 
         'journal', 'preproof', 'introduction', 'license', 'authorfunder', 'funder', 'right', 'reuse', 'publish',
         'peerreviewed', 'peer', 'review', 'abstract', 'free', 'text', 'grant', 'certify', 'reserve',
         'institutional', 'affiliations', 'neutral', 'regard', 'jurisdictional', 'background',
         'attribution', 'distribution', 'please', 'cite', 'article', 'paper', 'report', 'perpetuity', 'perpetuitythe', 
          'permission', 'permit', 'version', 'result', 'study', 'research', 'design', 'method',
         'well', 'even','due', 'also', 'although', 'thus', 'may', 'could', 'would', 'however', 'properly', 'behalf',
         'respectively', 'et', 'al', 'de', 'la', 'le', 'ie', 'en', 'eg', 'ml', 'wa', 'ha', 'str', 'us']

df['abstract_processed'] = df['abstract_processed'].apply(utils_text.remove_stopwords, extra_stopwords=words)

# Fill this new label column with False
df['is_transmi_article'] = False

df.to_csv('data/cord19_non_transmission_processed.csv', index=False)
print(df.shape)
