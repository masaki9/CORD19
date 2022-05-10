'''
utils_vis.py : Collection of Visualization Utility Functions
'''

import numpy as np

def add_bar_value_labels(ax, spacing=5, decimal=4, size=10):
    ''' Add data labels to bar charts. 
    :param: ax: axes on the current figure
    :param: spacing: amount of space above each bar
    :param: decimal: number of decimals to be displayed
    :param: size: data label size '''
    # For each bar, place a label
    for rect in ax.patches:
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2
        data_label = np.round(rect.get_height(), decimals=decimal)
        ax.annotate(data_label, (x_value, y_value), xytext=(0, spacing), size=size,
                    textcoords='offset points', ha='center', va='bottom')
