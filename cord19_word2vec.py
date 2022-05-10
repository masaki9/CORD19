'''
cord19_word2vec.py : Perform Word2Vec for COVID-19 transmission related articles
'''

from gensim.models import Word2Vec
import pandas as pd
import utils_text as utils_text

pd.options.display.max_rows = 200
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)


if __name__ == "__main__":
    data = 'data/cord19_transmission_processed.csv'
    df = pd.read_csv(data, header=0, sep=',')
    texts = utils_text.process_texts_for_gensim(df['abstract_processed'])

    # Word2Vec Model
    w2v_model = Word2Vec(sentences=texts, window=5, min_count=10, workers=8, epochs=20)
    vectors = w2v_model.wv

    print("\nAirborne Transmission")
    print(vectors.most_similar('airborne_transmission', topn=10))
    
    print("\nVertical Transmission")
    print(vectors.most_similar('vertical_transmission', topn=10))
    
    print("\nHousehold Transmission")
    print(vectors.most_similar('household_transmission', topn=10))
    
    print("\nOmicron Variant")
    print(vectors.most_similar('omicron_variant', topn=10))
