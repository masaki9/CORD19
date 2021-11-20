from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch, helpers
import json
import os


os.chdir('airflow/dags/cord19') # Set the current working directory


def _ingest_data():
    # today = datetime.today().strftime('%Y-%m-%d')
    today = '2020-03-27' # Test

    try:
        with open(f'data/{today}/cleaned_articles.json') as json_docs_file:
            json_docs = json.load(json_docs_file)

        es = Elasticsearch("http://localhost:9200", api_key=Variable.get('ES_API_KEY'))
        response = helpers.bulk(es, json_docs, index = "cord19")
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
    dag_id='ingest_cord19_data',
    schedule_interval='30 23 * * *',
    catchup=False,
    tags=['cord19', 'data ingestion'],
    description='CORD19 Data Ingestion',
    default_args=default_args
) as dag:
    ingest_data_task = PythonOperator(
        task_id='cord19_data_ingestion', 
        python_callable=_ingest_data
    )

    ingest_data_task
