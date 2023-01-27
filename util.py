import torch
from collections import OrderedDict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def colorize(words, color_array):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('Purples')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for word, color in zip(words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        colored_string += template.format(color, ' ' + word + ' ')
    return HTML(colored_string)

def colorize_list(wordss, color_arrays):
    # words is a list of lists of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('Purples')
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for row in range(len(wordss)):
        for word, color in zip(wordss[row], color_arrays[row]):
            color = matplotlib.colors.rgb2hex(cmap(color)[:3])
            colored_string += template.format(color, ' ' + word + ' ')
        colored_string += '<br>'
    return HTML(colored_string)