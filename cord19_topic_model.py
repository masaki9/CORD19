'''
cord19_topic_model.py : Perform topic modeling for COVID-19 transmission related articles
'''

from gensim import corpora
from gensim.models import CoherenceModel, LdaModel, LdaMulticore, Phrases
from gensim.models.phrases import Phraser, FrozenPhrases
import matplotlib.pyplot as plt
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import utils_text as utils_text
import utils_vis as utils_vis

pd.options.display.max_rows = 200
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)


def compute_coherence_scores(corpus, dictionary, texts, start=2, end=10, random_state=100,
                             chunksize=10000, passes=10, iterations=100, per_word_topics=True, eval_every=None):
    ''' Evaluate coherence scores for LDA models 
    :param: corpus: list of lists of word IDs and their frequencies
    :param: dictionary: mapping between words and their IDs
    :param: texts: list of lists of texts
    :return: LDA topic models and coherence scores '''
    coherence_scores = []
    models = []

    for num_topics in range(start, end):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=id2word,
                         random_state=random_state, chunksize=chunksize, passes=passes,
                         iterations=iterations, per_word_topics=per_word_topics, eval_every=eval_every)
        models.append(model)
        
        cm = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append(cm.get_coherence())

    return models, coherence_scores


def plot_coherence_scores(coherence_scores, start, end):
    ''' Plot LDA coherence scores 
    :param: coherence_scores: list of coherence scores
    :param: start: start number of topics 
    :param: end: end number of topics '''
    plt.figure(figsize=(20, 12))
    x_axis = range(start, end)
    # Convert range of int to list of str not to show decimal places in x axis
    x_axis = list(map(str, x_axis))
    plt.bar(x_axis, coherence_scores, color='lightblue')
    plt.plot(x_axis, coherence_scores, color='lightpink')
    plt.xlabel('Number of Topics', size=14)
    plt.ylabel('Coherence Score', size=14)
    plt.title('LDA Coherence Scores', size=18)
    ax = plt.gca()
    utils_vis.add_bar_value_labels(ax)
    plt.savefig('output/coherence_scores.png')
    # plt.show()


if __name__ == "__main__":
    data = 'data/cord19_transmission_processed.csv'
    df = pd.read_csv(data, header=0, sep=',')

    texts = utils_text.process_texts_for_gensim(df['abstract_processed'])

    # Create dictionary
    id2word = corpora.Dictionary(texts)

    # Filter out words that occur in less than 20 documents or more than 50% of the documents
    id2word.filter_extremes(no_below=20, no_above=0.5)

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

    random_state = 16
    chunksize = 10000
    passes = 40
    iterations = 400
    per_word_topics = True
    eval_every = None

    start, end = 2, 19
    models, scores = compute_coherence_scores(corpus=corpus, dictionary=id2word, texts=texts, start=start, end=end,
                                      random_state=random_state, chunksize=chunksize, passes=passes,
                                      iterations=iterations, per_word_topics=per_word_topics, eval_every=eval_every)
    print(scores)
    print("The highest coherence score is: {}".format(max(scores)))

    plot_coherence_scores(scores, start=start, end=end)

    lda_model = models[scores.index(max(scores))] # model with the highest score
    
    # # Build LDA model
    # num_topics = 9
    # lda_model = LdaModel(corpus=corpus,
    #                      id2word=id2word,
    #                      num_topics=num_topics, 
    #                      random_state=random_state,
    #                      chunksize=chunksize,
    #                      passes=passes,
    #                      iterations=iterations,
    #                      per_word_topics=per_word_topics,
    #                      eval_every=eval_every)
    # # Evaluate the LDA model
    # cm = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
    # score = cm.get_coherence()
    # print("Coherence Score: {}".format(score))

    # Print all topics (num_topics=-1) in the model (ordered by significance)
    for num, keyword in lda_model.print_topics(num_topics=-1, num_words=10):
        print('Topic {}: {}'.format(num + 1, keyword))

    # Visualize the topics
    lda_vis_prepared = gensimvis.prepare(topic_model=lda_model, corpus=corpus, dictionary=id2word, sort_topics=False)
    pyLDAvis.save_html(lda_vis_prepared, 'output/lda_topic_model.html')
    # pyLDAvis.show(lda_vis_prepared, local=False)  #  If False, use the standard urls.
