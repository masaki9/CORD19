from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from urllib.request import urlretrieve
import shutil
import os


os.chdir('airflow/dags/cord19') # Set the current working directory

# today = datetime.today().strftime('%Y-%m-%d')
today = '2020-03-27' # Test


def _download_data():
    if not os.path.exists('data'):
        os.makedirs('data')

    try:
        url = 'https://ai2-semanticscholar-cord-19.s3-us-west-2.amazonaws.com/historical_releases/cord-19_' + today + '.tar.gz'
        dst = 'data/cord19.tar.gz'
        response = urlretrieve(url, dst)
        response = 'Downloaded {}'.format(response[0])
    except Exception as err:
        response = err

    return response


def _decompress_data():
    try:
        file_path = 'data/cord19.tar.gz'
        extract_path = 'data'

        # Extract the downloaded file
        shutil.unpack_archive(file_path, extract_path)

        extracted_data_path = extract_path + '/' + today

        # Extract all tar.gz files in the extracted data folder
        for file_name in os.listdir(extracted_data_path):
            if file_name.endswith('.tar.gz'):
                print(file_name)
                file_to_extract = extracted_data_path + '/' + file_name
                shutil.unpack_archive(file_to_extract, extracted_data_path)
            
        response = 'Success'
    except Exception as err:
        response = err
   
    return response


with DAG(
    dag_id='download_cord19_data',
    schedule_interval='30 23 * * *',
    start_date=datetime(2021, 1, 1),
    catchup=False,
    dagrun_timeout=timedelta(minutes=60),
    tags=['cord19', 'data download'],
    description='CORD19 Data Download',
) as dag:
    download_data_task = PythonOperator(
        task_id='download_data',
        python_callable=_download_data
    )

    decompress_data_task = PythonOperator(
        task_id='decompress_data',
        python_callable=_decompress_data
    )

    download_data_task >> decompress_data_task
