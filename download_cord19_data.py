from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from urllib.request import urlretrieve
import shutil
import os
import pendulum


os.chdir('airflow') # Set the current working directory

date = datetime.today().strftime('%Y-%m-%d')


def download_data():
    if not os.path.exists('data'):
        os.makedirs('data')

    # try:
    url = 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_' + date + '.tar.gz'
    dst = 'data/cord19.tar.gz'
    response = urlretrieve(url, dst)
    response = 'Downloaded {}'.format(response[0])
    # except Exception as err:
    #     response = err

    return response


def decompress_data():
    file_path = 'data/cord19.tar.gz'
    extract_path = 'data'

    # Extract the downloaded file
    shutil.unpack_archive(file_path, extract_path)

    extracted_data_path = extract_path + '/' + date

    # Extract all tar.gz files in the extracted data folder
    for file_name in os.listdir(extracted_data_path):
        if file_name.endswith('.tar.gz'):
            file_to_extract = extracted_data_path + '/' + file_name
            shutil.unpack_archive(file_to_extract, extracted_data_path)
        
    response = f'Extracted data in {extracted_data_path}'

    return response


with DAG(
    dag_id='download_cord19_data',
    schedule_interval='01 21 * * *',
    tags=['cord19', 'data download'],
    description='CORD19 Data Download',
    start_date=pendulum.datetime(2021, 1, 1, tz='local'),
    catchup=False
) as dag:
    download_data_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
        execution_timeout=timedelta(seconds=2400)
    )

    decompress_data_task = PythonOperator(
        task_id='decompress_data',
        python_callable=decompress_data,
        execution_timeout=timedelta(seconds=2400)
    )

    download_data_task >> decompress_data_task
