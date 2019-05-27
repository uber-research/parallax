# Copyright (c) 2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import re
import sys

import bokeh
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import layout, widgetbox
from bokeh.models import Div
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.plotting import figure

from globals import default_test_datasets, default_embeddings_top_k
from modules.api import projection
from modules.data_manager import DataManager

PI = np.pi
default_color = 'steelblue'
highlight_color = 'tomato'
colors = bokeh.palettes.Category10[10]

num_axes = 0
theta = np.linspace(PI * 2, 0, num_axes, endpoint=False) + PI / 2


def polar_chart_vertices(theta, radius=1):
    """
    return vertices of the polar axes centered at [0.5, 0.5] with radius = 0.5
    """
    return [(radius * np.cos(t), radius * np.sin(t)) for t in theta]


def formula_patch(value, theta):
    """
    return vertices of a formula patch
    """
    dx, dy = value * np.cos(theta), value * np.sin(theta)
    return dx, dy


def reset_plot():
    """
    iterate through all visual element and set them to invisible
    """
    global visual_elements
    if len(visual_elements) > 0:
        for el in visual_elements:
            el.visible = False


def select_embeddings():
    axes_list = []
    axes_val = axes.value.strip()
    if axes_val != '':
        axes_list = re.split("\s*;\s*", axes_val)

    items_list = []
    items_val = items.value.strip()
    if items_val != '':
        items_list = re.split("\s*;\s*", items_val)

    if len(axes_list) >= 3 and len(items_list) >= 1:

        '''
        if projection_tab_panel.active == 0:  # explicit
            mode = 'explicit'
            metric = measure_1.value
            pre_filtering = True
            post_filtering = False
        elif projection_tab_panel.active == 1:  # pca
            mode = 'pca'
            metric = None
            formulae = None
            pre_filtering = False
            post_filtering = True
        else:  # if projection_tab_panel.active == 2:  # tsne
            mode = 'tsne'
            metric = measure_3.value
            formulae = None
            pre_filtering = False
            post_filtering = True
        '''

        return projection(data_manager,
                          dataset_id=dataset.value,
                          mode='explicit',
                          rank_slice=(0, 0),
                          metric=measure.value,
                          formulae=axes_list,
                          items=items_list,
                          pre_filtering=True,
                          post_filtering=False,
                          )
    else:
        return {}


def update(attr, old, new):
    reset_plot()

    axes_list = []
    axes_val = axes.value.strip()
    if axes_val != '':
        axes_list = re.split("\s*;\s*", axes_val)

    num_axes = len(axes_list)
    theta = np.linspace(PI * 2, 0, num_axes, endpoint=False) + PI / 2

    embeddings = select_embeddings()
    plot.title.text = "%d embeddings selected" % len(embeddings)

    pdfs = []
    i = 0
    for label, emb_dict in embeddings.items():
        coords = emb_dict['coords']
        coords = [max(coord, 0) for coord in coords]
        pdfs.append(coords)
        x, y = formula_patch(coords, theta)
        patch = plot.patch(x=x, y=y, fill_alpha=opacity.value,
                           fill_color=colors[i % len(colors)], legend=label,
                            line_color=colors[i % len(colors)], line_alpha=1, line_width=3)
        patch_label = plot.text(
            x=x, y=y, text=["%.2f" % n for n in emb_dict['coords']],
            text_baseline="middle", text_align="center", text_font_size='9pt')
        visual_elements.append(patch)
        visual_elements.append(patch_label)
        i += 1

    radius = 1 if len(pdfs) == 0 else min(1, max(np.array(pdfs).flatten()) * 1.05)
    vertices = polar_chart_vertices(theta, radius)
    x_v = [v[0] for v in vertices]
    y_v = [v[1] for v in vertices]

    for v in vertices:
        xs = [0, v[0]]
        ys = [0, v[1]]
        line = plot.line(x=xs, y=ys)
        label = plot.text(x=x_v, y=y_v, text=axes_list)
        visual_elements.append(line)
        visual_elements.append(label)

        # ticks: # segments, 6 ticks
        num_ticks = 6
        xticks = np.linspace(0, v[0], num_ticks)
        yticks = np.linspace(0, v[1], num_ticks)
        circle = plot.circle(x=xticks, y=yticks, color="gray", alpha=0.5)
        visual_elements.append(circle)

    plot.legend.location = "top_left"
    plot.legend.click_policy = "hide"


# I/O
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=json.loads, default=json.dumps(default_test_datasets),
                    help='loads custom embeddings. It accepts a JSON string containing a list of dictionaries. '
                         'Each dictionary should contain a name field, an embedding_file filed '
                         'and a metadata_file field. '
                         'For example: \'[{"name": "wikipedia", "embedding_file": "...", "metadata_file": "..."}, '
                         '{"name": "twitter", "embedding_file": "...", "metadata_file": "..."}]\'')
parser.add_argument('-k', '--first_k', type=int, default=default_embeddings_top_k,
                    help='loads only the first k embeddings from the mebeddings files. -1 mean load them all')
parser.add_argument('-l', '--labels', action='store_true', default=False,
                    help='show labels')
parser.add_argument('-o', '--output_backend', default='webgl', choices=['webgl', 'canvas', 'svg'],
                    help='backend to use for rendering. webgl is the fastest, only saves in PNG format, '
                         'svg should be used for saving in SVG format, canvas is the fallback option')
args = parser.parse_args(sys.argv[1:])
data_manager = DataManager(args.datasets, args.first_k)
selected_dataset = data_manager.dataset_ids[0]

# CONTROLS
dataset = Select(title="Dataset", options=data_manager.dataset_ids, value=selected_dataset)
measure = Select(title="Measure",
                 options=[('cosine_similarity', 'Cosine Similarity'),
                          ('cosine', 'Cosine Distance'),
                          ('euclidean', 'Euclidean'),
                          ('dot_product', 'Dot product'),
                          ('correlation', 'Correlation'),
                          ],
                 value='cosine_similarity')
axes_preset = '; '.join(['usa', 'europe', 'china', 'japan', 'brazil'])
axes = TextInput(title='Axes Formulae (separated by ";")', placeholder=axes_preset)

items_preset = '; '.join(['food', 'movie'])
items = TextInput(title='Items Formulae (separated by ";")', placeholder=items_preset)

visualization_title = Div(text="<strong>Visualization</strong>")
opacity = Slider(title="Opacity", value=0.2, start=0, end=1, step=0.1)

# BIND CONTROL EVENTS
dataset.on_change('value', update)
axes.on_change('value', update)
items.on_change('value', update)
measure.on_change('value', update)
opacity.on_change('value', update)

# GLOBAL VIEW COMPONENT
plot = figure(plot_height=640, plot_width=640, output_backend=args.output_backend)
visual_elements = []

controls = [dataset, measure, axes, items, visualization_title, opacity]
control_panel = widgetbox(children=controls)

curdoc().add_root(layout([[plot, control_panel]]))
curdoc().title = "Polar"
