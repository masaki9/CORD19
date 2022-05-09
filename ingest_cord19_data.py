from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers
import glob
import json
import os
import pendulum


os.chdir('airflow') # Set the current working directory


def ingest_data():
    date = datetime.today().strftime('%Y-%m-%d')

    json_filenames = glob.glob(f'data/{date}/cleaned_articles_*.json', recursive=False)

    nums_ingested = []
    for i in json_filenames:
        with open(i) as json_docs_file:
            json_docs = json.load(json_docs_file)

        es = Elasticsearch('http://localhost:9200', api_key=Variable.get('ES_API_KEY'))
        res = helpers.bulk(es, json_docs, index="cord19")
        nums_ingested.append(res[0])

    response = f'Ingested {sum(nums_ingested)} documents'

    return response


with DAG(
    dag_id='ingest_cord19_data',
    schedule_interval='31 23 * * *',
    tags=['cord19', 'data ingestion'],
    description='CORD19 Data Ingestion',
    start_date=pendulum.datetime(2021, 1, 1, tz='local'),
    catchup=False
) as dag:
    ingest_data_task = PythonOperator(
        task_id='cord19_data_ingestion',
        python_callable=ingest_data,
        execution_timeout=timedelta(seconds=2400)
    )

    ingest_data_task
