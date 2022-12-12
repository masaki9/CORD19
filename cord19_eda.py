'''
cord19_eda.py : Visualize COVID-19 transmission related articles for exploratory data analysis
'''

import matplotlib.pyplot as plt
import pandas as pd
import utils_text as utils_text
import utils_vis as utils_vis
from wordcloud import WordCloud

pd.options.display.max_rows = 200
pd.set_option('display.max_rows', 300)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)
pd.set_option('display.max_colwidth', 80)

data = 'data/cord19_transmission_processed.csv'
df = pd.read_csv(data, header=0, sep=',')

plt.figure(figsize=(20, 12))
wc = WordCloud(height=2000, width=2000, background_color='lightgrey', colormap='copper_r')
wc = wc.generate(' '.join(df['abstract_processed']))
plt.imshow(wc, interpolation='bilinear')
# plt.title('Common Words in COVID-19 Transmission Related Articles', size=18)
plt.axis('off')
# plt.show()
plt.savefig('output/wordcloud.png')

ngrams = open('output/ngrams.txt', 'w')

# Unigrams
unigrams = utils_text.get_ngrams_df(df, 'abstract_processed', 1)
print('Unigrams\n{}'.format(unigrams.head(n=300)), file=ngrams)

plt.figure(figsize=(20, 12))
x_axis = unigrams['N-Gram'][:20]
y_axis = unigrams['Frequency'][:20]
plt.bar(x_axis, y_axis, color='sienna')
plt.xlabel('Unigram', size=14)
plt.xticks(rotation=15)
plt.ylabel('Frequency', size=14)
plt.title('Top 20 Unigrams Related to COVID-19 Transmission', size=18)
ax = plt.gca()
utils_vis.add_bar_value_labels(ax)
# plt.show()
plt.savefig('output/top20_unigrams.png')


# Bigrams
bigrams = utils_text.get_ngrams_df(df, 'abstract_processed', 2)
print('\nBigrams\n{}'.format(bigrams.head(n=300)), file=ngrams)

plt.figure(figsize=(20, 12))
x_axis = bigrams['N-Gram'][:20]
y_axis = bigrams['Frequency'][:20]
plt.bar(x_axis, y_axis, color='sienna')
plt.xlabel('Bigram', size=14)
plt.xticks(rotation=30)
plt.ylabel('Frequency', size=14)
plt.title('Top 20 Bigrams Related to COVID-19 Transmission', size=18)
ax = plt.gca()
utils_vis.add_bar_value_labels(ax)
# plt.show()
plt.savefig('output/top20_bigrams.png')


# Trigrams
trigrams = utils_text.get_ngrams_df(df, 'abstract_processed', 3)
print('\nTrigrams\n{}'.format(trigrams.head(n=300)), file=ngrams)

plt.figure(figsize=(20, 12))
x_axis = trigrams['N-Gram'][:20]
y_axis = trigrams['Frequency'][:20]
plt.bar(x_axis, y_axis, color='sienna')
plt.xlabel('Trigram', size=14)
plt.xticks(rotation=30)
plt.ylabel('Frequency', size=14)
plt.title('Top 20 Trigrams Related to COVID-19 Transmission', size=18)
ax = plt.gca()
utils_vis.add_bar_value_labels(ax)
plt.gcf().subplots_adjust(bottom=0.15)
# plt.show()
plt.savefig('output/top20_trigrams.png')


# 4-grams
four_grams = utils_text.get_ngrams_df(df, 'abstract_processed', 4)
print('\n4-grams\n{}'.format(four_grams.head(n=50)), file=ngrams)

# 5-grams
five_grams = utils_text.get_ngrams_df(df, 'abstract_processed', 5)
print('\n5-grams\n{}'.format(five_grams.head(n=50)), file=ngrams)

# 6-grams
six_grams = utils_text.get_ngrams_df(df, 'abstract_processed', 6)
print('\n6-grams\n{}'.format(six_grams.head(n=50)), file=ngrams)

ngrams.close()
