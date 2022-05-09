from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import glob
import json
import numpy as np
import os
import pandas as pd
import pendulum


os.chdir('airflow') # Set the current working directory


def process_articles(json_files):
    paper_ids, titles, abstracts, text_bodies = [], [], [], []

    for json_file in json_files:
        with open(json_file) as json_data:
            data = json.load(json_data)

            paper_ids.append(data['paper_id'])
            titles.append(data['metadata']['title'])

            abstract_texts = [x['text'] for x in data['abstract']]
            abstract = '\n'.join(abstract_texts)
            abstracts.append(abstract)

            texts = [(x['section'], x['text']) for x in data['body_text']] # Extract only sections and texts

            section_text_dict = {x['section']: '' for x in data['body_text']} # Dict of sections and empty texts
            for section, text in texts:
                section_text_dict[section] += text

            # Format the body before adding it to the list
            body = ''
            for section, text in section_text_dict.items():
                body += section + '\n'
                body += text + '\n\n'
            
            text_bodies.append(body)
    
    df = pd.DataFrame({'_id': paper_ids, 'paper_id': paper_ids, 'title': titles, 'abstract': abstracts, 'text_body': text_bodies})
    return df


def clean_data():
    date = datetime.today().strftime('%Y-%m-%d')
    
    json_filenames = glob.glob(f'data/{date}/document_parses/pdf_json/*.json', recursive=False)
    df = process_articles(json_filenames)

    # Perform a left join and add metadata to df
    df_metadata = pd.read_csv(f'data/{date}/metadata.csv', low_memory=False)
    df_metadata = df_metadata[['sha', 'publish_time', 'authors', 'source_x', 'url']]
    df = pd.merge(df, df_metadata, left_on='paper_id', right_on='sha', how='left')
    df.drop('sha', axis=1, inplace=True)

    # Split data into multiple files not to overwhelm ES during indexing
    number_of_chunks = 12
    for i, chunk in enumerate(np.array_split(df, number_of_chunks)):
        chunk.to_json(f'data/{date}/cleaned_articles_{i}.json', orient='records')

    response = 'Successfully cleaned data'
   
    return response


with DAG(
    dag_id='clean_cord19_data',
    schedule_interval='31 22 * * *',
    tags=['cord19', 'data cleaning'],
    description='CORD19 Data Cleaning',
    start_date=pendulum.datetime(2021, 1, 1, tz='local'),
    catchup=False
) as dag:
    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
        execution_timeout=timedelta(seconds=2400)
    )

    clean_data_task
