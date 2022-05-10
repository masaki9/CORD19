'''
cord19_etl_trans.py : Extract, transform, and load COVID-19 transmission related articles
'''

from elasticsearch import Elasticsearch, helpers
import os
import pandas as pd
import utils_text as utils_text

es = Elasticsearch('http://localhost:9200', api_key=os.environ['ES_API_KEY'])

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
                        "must" : {
                            "regexp": {
                                "title" : ".*transmi.*"
                            }
                        }
                    }
                },
                {
                    "bool": {
                        "must" : {
                            "regexp": {
                                "abstract" : ".*transmi.*"
                            }
                        }
                    }
                }
            ]
        }
    }
}
results = helpers.scan(es, query=body, index='cord19', scroll='5m', preserve_order=True)

df = pd.DataFrame.from_dict([document['_source'] for document in results])
print(df.shape)
print(df.head())

df = df[["paper_id","title", "abstract", "text_body"]]
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

# Fill this new label column with True
df['is_transmi_article'] = True

df.to_csv('data/cord19_transmission_processed.csv', index=False)
print(df.shape)
