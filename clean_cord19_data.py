from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import glob
import json
import os
import pandas as pd


os.chdir('airflow/dags/cord19') # Set the current working directory


def _process_articles(json_files):
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


def _clean_data():
    try:
        # today = datetime.today().strftime('%Y-%m-%d')
        today = '2020-03-27' # Test
        json_filenames = glob.glob(f'data/{today}/*/*.json', recursive=False)
        df = _process_articles(json_filenames)

        # Perform a left join and add metadata to df
        df_metadata = pd.read_csv(f'data/{today}/metadata.csv')
        df_metadata = df_metadata[['sha', 'publish_time', 'authors', 'source_x', 'url']]
        df = pd.merge(df, df_metadata, left_on='paper_id', right_on='sha', how='left')
        df.drop('sha', axis=1, inplace=True)

        df.to_json(f'data/{today}/cleaned_articles.json', orient='records')

        response = 'Successfully cleaned data'
    except Exception as err:
        response = err
   
    return response


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2021, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='clean_cord19_data',
    schedule_interval='30 23 * * *',
    start_date=datetime(2021, 1, 1),
    catchup=False,
    dagrun_timeout=timedelta(minutes=60),
    tags=['cord19', 'data cleaning'],
    description='CORD19 Data Cleaning',
    default_args=default_args
) as dag:
    clean_data_task = PythonOperator(
        task_id='clean_data',
        python_callable=_clean_data
    )

    clean_data_task

