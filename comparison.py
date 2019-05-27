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

import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import layout, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div, LabelSet, \
    RangeSlider, MultiSelect, Line, Button, Segment
from bokeh.models.widgets import Slider, Select, TextInput
from bokeh.plotting import figure

from globals import default_test_datasets, default_embeddings_top_k
from modules.api import projection
from modules.data_manager import DataManager

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datasets', type=json.loads,
                    default=json.dumps(default_test_datasets),
                    help='loads custom embeddings. It accepts a JSON string containing a list of dictionaries. '
                         'Each dictionary should contain a name field, an embedding_file filed '
                         'and a metadata_file field. '
                         'For example: \'[{"name": "wikipedia", "embedding_file": "...", "metadata_file": "..."}, '
                         '{"name": "twitter", "embedding_file": "...", "metadata_file": "..."}]\'')
parser.add_argument('-k', '--first_k', type=int,
                    default=default_embeddings_top_k,
                    help='loads only the first k embeddings from the mebeddings files. -1 mean load them all')
parser.add_argument('-l', '--labels', action='store_true', default=False,
                    help='show labels')
parser.add_argument('-o', '--output_backend', default='webgl',
                    choices=['webgl', 'canvas', 'svg'],
                    help='backend to use for rendering. webgl is the fastest, only saves in PNG format, '
                         'svg should be used for saving in SVG format, canvas is the fallback option')
args = parser.parse_args(sys.argv[1:])

default_color_1 = 'steelblue'
default_color_2 = 'orange'
default_segment_color = 'black'
# highlight_color_1 = 'indigo'
# highlight_color_2 = 'orangered'
# highlight_segment_color = 'darkred'
highlight_color_1 = 'steelblue'
highlight_color_2 = 'orange'
highlight_segment_color = 'grey'

data_manager = DataManager(args.datasets, args.first_k)
if data_manager.get_num_datasets() < 2:
    print("The comparison view need at least two datasets.")
    sys.exit(-1)
selected_dataset_1 = data_manager.dataset_ids[0]
selected_dataset_2 = data_manager.dataset_ids[1]

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x0=[], y0=[], x1=[], y1=[], color=[]))
source_1 = ColumnDataSource(
    data=dict(x=[], y=[], color=[], label=[], legend=[]))
source_2 = ColumnDataSource(
    data=dict(x=[], y=[], color=[], label=[], legend=[]))
line_source = ColumnDataSource(dict(x=[], y=[]))

hover = HoverTool(tooltips=[
    ("Label", "@label"),
    ("x", "@x"),
    ("y", "@y")
],
    names=["scatter", "scatter_1", "scatter_2"])

plot = figure(plot_height=800, plot_width=800, title="",
              toolbar_location='right', output_backend=args.output_backend)
plot.add_tools(hover)
line = Line(x='x', y='y', line_width=2)
plot.add_glyph(line_source, line)
scatter_1 = plot.circle(x="x", y="y", source=source_1, size=7, color="color",
                        line_color=None, fill_alpha=0.6,
                        name='scatter_1',
                        legend='legend')
scatter_2 = plot.circle(x="x", y="y", source=source_2, size=7, color="color",
                        line_color=None, fill_alpha=0.6,
                        name='scatter_2',
                        legend='legend')
segments = Segment(x0="x0", y0="y0", x1="x1", y1="y1", line_color="color",
                   line_alpha=0.6, line_width=2,
                   line_dash='dashed',
                   name='scatter')
plot.add_glyph(source, segments)
plot.legend.location = "top_left"
plot.legend.click_policy = "hide"
if args.labels:
    labels_annotations_1 = LabelSet(x="x", y="y", text="label", y_offset=1,
                                    text_font_size="8pt", text_color="color",
                                    source=source_1, text_align='center',
                                    text_alpha=1)
    plot.add_layout(labels_annotations_1)
    labels_annotations_2 = LabelSet(x="x", y="y", text="label", y_offset=1,
                                    text_font_size="8pt", text_color="color",
                                    source=source_2, text_align='center',
                                    text_alpha=1)
    plot.add_layout(labels_annotations_2)

# Create Input controls
dataset_1 = Select(title="Dataset #1", options=data_manager.dataset_ids,
                   value=selected_dataset_1)
dataset_2 = Select(title="Dataset #2", options=data_manager.dataset_ids,
                   value=selected_dataset_2)

measure = Select(title="Measure",
                 options=[('cosine_similarity', 'Cosine Similarity'),
                          ('cosine', 'Cosine Distance'),
                          ('euclidean', 'Euclidean'),
                          ('dot_product', 'Dot product'),
                          ('correlation', 'Correlation'),
                          ],
                 value='cosine_similarity')
x_axis = TextInput(title="X Axis Formula", placeholder="formula")
y_axis = TextInput(title="Y Axis Formula", placeholder="formula")
items = TextInput(title='Items Filter Formulae (separated by ";")',
                  placeholder="formula, formula, ...")

metadata_filters_title_1 = Div(
    text="<strong>Metadata Filters Dataset 1</strong>")

rank_slice_1 = RangeSlider(title="Rank Slice",
                           value=(1, data_manager.get_size(selected_dataset_1)),
                           start=1,
                           end=data_manager.get_size(selected_dataset_1),
                           step=1)

metadata_type_1 = data_manager.get_metadata_type(selected_dataset_1)
metadata_domain_1 = data_manager.get_metadata_domain(selected_dataset_1)

metadata_filters_1 = []
for attribute in metadata_type_1:
    m_type = metadata_type_1[attribute]
    m_domain = metadata_domain_1[attribute]
    if m_type == 'boolean':
        filter = Select(title=attribute, value="Any",
                        options=["Any", "True", "False"])
    elif m_type == 'numerical':
        filter = RangeSlider(start=m_domain[0], end=m_domain[1],
                             value=m_domain, step=1, title=attribute)
    elif m_type == 'categorical':
        categories = sorted(list(metadata_domain_1[attribute]))
        filter = MultiSelect(title=attribute, value=categories,
                             options=categories)
    elif m_type == 'set':
        categories = sorted(list(metadata_domain_1[attribute]))
        filter = MultiSelect(title=attribute, value=categories,
                             options=categories)
    else:
        raise ValueError(
            'Unsupported attribute type {} in metadata'.format(m_type))
    metadata_filters_1.append(filter)

metadata_filters_title_2 = Div(
    text="<strong>Metadata Filters Dataset 2</strong>")

rank_slice_2 = RangeSlider(title="Rank Slice",
                           value=(1, data_manager.get_size(selected_dataset_2)),
                           start=1,
                           end=data_manager.get_size(selected_dataset_2),
                           step=1)

metadata_type_2 = data_manager.get_metadata_type(selected_dataset_2)
metadata_domain_2 = data_manager.get_metadata_domain(selected_dataset_2)

metadata_filters_2 = []
for attribute in metadata_type_1:
    m_type = metadata_type_2[attribute]
    m_domain = metadata_domain_2[attribute]
    if m_type == 'boolean':
        filter = Select(title=attribute, value="Any",
                        options=["Any", "True", "False"])
    elif m_type == 'numerical':
        filter = RangeSlider(start=m_domain[0], end=m_domain[1],
                             value=m_domain, step=1, title=attribute)
    elif m_type == 'categorical':
        categories = sorted(list(metadata_domain_2[attribute]))
        filter = MultiSelect(title=attribute, value=categories,
                             options=categories)
    elif m_type == 'set':
        categories = sorted(list(metadata_domain_2[attribute]))
        filter = MultiSelect(title=attribute, value=categories,
                             options=categories)
    else:
        raise ValueError(
            'Unsupported attribute type {} in metadata'.format(m_type))
    metadata_filters_2.append(filter)

data_filters_title = Div(text="<strong>Data Filters</strong>")

difference_slider = RangeSlider(start=0, end=2, value=(0, 2), step=0.01,
                                title='Difference Filter')
slope_slider = RangeSlider(start=-100, end=100, value=(-100, +100), step=0.01,
                           title='Slope Filter')


def build_data_filter():
    measure_cf = Select(options=[('cosine_similarity', 'Cosine Similarity'),
                                 ('cosine', 'Cosine Distance'),
                                 ('euclidean', 'Euclidean'),
                                 ('dot_product', 'Dot product'),
                                 ('correlation', 'Correlation'),
                                 ],
                        value='cosine_similarity')
    formula_cf = TextInput(value='', placeholder="formula")
    compare_cf = Select(options=[('greater', '>'),
                                 ('greater_equal', '≥'),
                                 ('equal', '='),
                                 ('less_equal', '≤'),
                                 ('less', '<'),
                                 ],
                        value='greater', width=20)
    value_cf = TextInput(value='', placeholder="numeric value")
    dataset_cf = Select(title="Dataset", options=['Both', selected_dataset_1,
                                                  selected_dataset_2],
                        value='Both')
    data_filter = [measure_cf, formula_cf, compare_cf, value_cf, dataset_cf]
    return data_filter


data_filter = build_data_filter()
data_filters = [widget for widget in data_filter]
data_filters_groups = [data_filter]
add_data_filter_button = Button(label="Add", button_type="success")

visualization_title = Div(text="<strong>Visualization</strong>")

opacity = Slider(title="Opacity", value=0.6, start=0, end=1, step=0.01)
axes_font_size = Slider(title="Axes Font Size", value=8, start=8, end=32)
if args.labels:
    show_labels = Select(title="Show labels:", value="True",
                         options=["True", "False"])
    labels_font_size = Slider(title="Labels Font Size", value=8, start=8,
                              end=32)


def select_embeddings():
    x_axis_value = x_axis.value.strip()
    y_axis_value = y_axis.value.strip()
    if x_axis_value == '' or y_axis_value == '':
        return {}
    else:
        metadata_filters_params_1 = []
        for metadata_filter in metadata_filters_1:
            if metadata_type_1[metadata_filter.title] == 'boolean':
                if metadata_filter.value != 'Any':
                    metadata_filters_params_1.append((metadata_filter.title,
                                                      metadata_filter.value == 'True'))
            elif metadata_type_1[metadata_filter.title] == 'numerical':
                filter_value = (
                    int(metadata_filter.value[0]),
                    int(metadata_filter.value[1]))
                if filter_value != (
                        int(rank_slice_1.start), int(rank_slice_1.end)):
                    metadata_filters_params_1.append(
                        (metadata_filter.title, filter_value))
            elif metadata_type_1[metadata_filter.title] == 'categorical':
                filter_value = set(metadata_filter.value)
                if len(filter_value) > 0 and filter_value != metadata_domain_1[
                    metadata_filter.title]:
                    metadata_filters_params_1.append(
                        (metadata_filter.title, filter_value))
            elif metadata_type_1[metadata_filter.title] == 'set':
                filter_value = set(metadata_filter.value)
                if len(filter_value) > 0 and filter_value != metadata_domain_1[
                    metadata_filter.title]:
                    metadata_filters_params_1.append(
                        (metadata_filter.title, filter_value))

        metadata_filters_params_2 = []
        for metadata_filter in metadata_filters_2:
            if metadata_type_2[metadata_filter.title] == 'boolean':
                if metadata_filter.value != 'Any':
                    metadata_filters_params_2.append((metadata_filter.title,
                                                      metadata_filter.value == 'True'))
            elif metadata_type_2[metadata_filter.title] == 'numerical':
                filter_value = (
                    int(metadata_filter.value[0]),
                    int(metadata_filter.value[1]))
                if filter_value != (
                        int(rank_slice_2.start), int(rank_slice_2.end)):
                    metadata_filters_params_2.append(
                        (metadata_filter.title, filter_value))
            elif metadata_type_2[metadata_filter.title] == 'categorical':
                filter_value = set(metadata_filter.value)
                if filter_value != metadata_domain_2[metadata_filter.title]:
                    metadata_filters_params_2.append(
                        (metadata_filter.title, filter_value))
            elif metadata_type_2[metadata_filter.title] == 'set':
                filter_value = set(metadata_filter.value)
                if len(filter_value) > 0 and filter_value != metadata_domain_2[
                    metadata_filter.title]:
                    metadata_filters_params_2.append(
                        (metadata_filter.title, filter_value))

        data_filters_params = []
        for data_filter in data_filters_groups:
            formula = data_filter[1].value.strip()
            number_value = data_filter[3].value.strip()
            if formula != '' and number_value != '':
                measure_val = data_filter[0].value
                compare_function = data_filter[2].value
                try:
                    number = float(number_value)
                    data_filters_params.append({'measure': measure_val,
                                                'formula': formula,
                                                'compare_function': compare_function,
                                                'number': number})
                except:
                    print('invalid number value:', number_value)

        items_list = []
        items_val = items.value.strip()
        if items_val != '':
            items_list = re.split("\s*;\s*", items_val)

        mode = 'explicit'
        metric = measure.value
        formulae = [x_axis_value, y_axis_value]
        pre_filtering = True
        post_filtering = False

        rank_slice_values_1 = (
            int(rank_slice_1.value[0]), int(rank_slice_1.value[1]))
        if rank_slice_values_1 == (
                int(rank_slice_1.start), int(rank_slice_1.end)):
            rank_slice_values_1 = None

        rank_slice_values_2 = (
            int(rank_slice_2.value[0]), int(rank_slice_2.value[1]))
        if rank_slice_values_2 == (
                int(rank_slice_2.start), int(rank_slice_2.end)):
            rank_slice_values_2 = None

        embeddings_1 = projection(
            data_manager,
            dataset_id=dataset_1.value,
            data_filters=data_filters_params,
            metadata_filters=metadata_filters_params_1,
            mode=mode,
            rank_slice=rank_slice_values_1,
            metric=metric,
            n_axes=2,
            formulae=formulae,
            items=items_list,
            pre_filtering=pre_filtering,
            post_filtering=post_filtering,
        )

        embeddings_2 = projection(
            data_manager,
            dataset_id=dataset_2.value,
            data_filters=data_filters_params,
            metadata_filters=metadata_filters_params_2,
            mode=mode,
            rank_slice=rank_slice_values_2,
            metric=metric,
            n_axes=2,
            formulae=formulae,
            items=items_list,
            pre_filtering=pre_filtering,
            post_filtering=post_filtering,
        )

        embeddings = {}
        common_keys = set(embeddings_1.keys()).intersection(
            set(embeddings_2.keys()))

        max_difference = 0
        for key in common_keys:
            max_difference = max(max_difference, np.linalg.norm(
                embeddings_1[key]['coords'] - embeddings_2[key]['coords']))

        difference_slider._callbacks['value'] = []
        old_end = difference_slider.end
        if max_difference > 0:
            difference_slider.end = max_difference
        if len(common_keys) > 0:
            if difference_slider.value[1] == old_end:
                difference_slider.update(
                    value=(0, difference_slider.end))
            else:
                difference_slider.update(
                    value=(0, min(max_difference, difference_slider.value[1])))
        difference_slider.on_change('value', update)

        for key in common_keys:

            distance = np.linalg.norm(
                embeddings_1[key]['coords'] - embeddings_2[key]['coords'])
            slope = (embeddings_2[key]['coords'][1] -
                     embeddings_1[key]['coords'][1]) / \
                    (embeddings_2[key]['coords'][0] -
                     embeddings_1[key]['coords'][0])

            if distance >= difference_slider.value[0] and distance <= \
                    difference_slider.value[1] and \
                    slope >= slope_slider.value[0] and slope <= \
                    slope_slider.value[1]:
                if len(items_list) == 0:
                    embeddings[key] = {'coords_1': embeddings_1[key]['coords'],
                                       'coords_2': embeddings_2[key]['coords']}
                else:
                    if key in items_list:
                        embeddings[key] = {
                            'coords_1': embeddings_1[key]['coords'],
                            'coords_2': embeddings_2[key]['coords']}
        return embeddings


def update(attr, old, new):
    embeddings = select_embeddings()

    plot.xaxis.axis_label = x_axis.value
    plot.yaxis.axis_label = y_axis.value

    measure_name = ' '.join(measure.value.split('_'))
    plot.title.text = "{} embeddings selected, {} measure".format(
        len(embeddings), measure_name)

    items_list = []
    items_val = items.value.strip()
    if items_val != '':
        items_list = re.split("\s*;\s*", items.value.strip())
    highlight_items = set(items_list)

    max_difference = 0
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    labels = []
    colors = []
    colors_1 = []
    colors_2 = []
    for label, emb_dict in embeddings.items():
        x0.append(emb_dict['coords_1'][0])
        y0.append(emb_dict['coords_1'][1])
        x1.append(emb_dict['coords_2'][0])
        y1.append(emb_dict['coords_2'][1])
        max_difference = max(max_difference, np.linalg.norm(
            emb_dict['coords_1'] - emb_dict['coords_2']))
        labels.append(label)
        colors.append(
            highlight_segment_color if label in highlight_items else default_segment_color)
        colors_1.append(
            highlight_color_1 if label in highlight_items else default_color_1)
        colors_2.append(
            highlight_color_2 if label in highlight_items else default_color_2)

    source.data = dict(
        x0=x0,
        y0=y0,
        x1=x1,
        y1=y1,
        color=colors,
        label=labels,
    )
    source_1.data = dict(
        x=x0,
        y=y0,
        color=colors_1,
        label=labels,
        legend=[selected_dataset_1] * len(labels)
    )
    source_2.data = dict(
        x=x1,
        y=y1,
        color=colors_2,
        label=labels,
        legend=[selected_dataset_2] * len(labels)
    )

    if len(embeddings) > 0:
        line_min = min(min(x0), min(y0), min(x1), min(y1))
        line_max = max(max(x0), max(y0), max(x1), max(y1))
        line_source.data = dict(x=[line_min, line_max], y=[line_min, line_max])


def update_dataset(attr, old, new):
    # Dataset 1
    global selected_dataset_1
    selected_dataset_1 = dataset_1.value
    rank_slice_1.end = data_manager.get_size(selected_dataset_1)
    rank_slice_1.update(value=(1, data_manager.get_size(selected_dataset_1)))

    metadata_type_1 = data_manager.get_metadata_type(selected_dataset_1)
    metadata_domain_1 = data_manager.get_metadata_domain(selected_dataset_1)

    while len(metadata_filters_1) > 0:
        metadata_filters_1.pop()
    for attribute in metadata_type_1:
        m_type = metadata_type_1[attribute]
        m_domain = metadata_domain_1[attribute]
        if m_type == 'boolean':
            filter = Select(title=attribute, value="Any",
                            options=["Any", "True", "False"])
        elif m_type == 'numerical':
            filter = RangeSlider(start=m_domain[0], end=m_domain[1],
                                 value=m_domain, step=1, title=attribute)
        elif m_type == 'categorical':
            categories = sorted(list(metadata_domain_1[attribute]))
            filter = MultiSelect(title=attribute, value=categories,
                                 options=categories)
        elif m_type == 'set':
            categories = sorted(list(metadata_domain_1[attribute]))
            filter = MultiSelect(title=attribute, value=categories,
                                 options=categories)
        else:
            raise ValueError(
                'Unsupported attribute type {} in metadata'.format(m_type))
        metadata_filters_1.append(filter)

    # Dataset 2
    global selected_dataset_2
    selected_dataset_2 = dataset_2.value
    rank_slice_2.end = data_manager.get_size(selected_dataset_2)
    rank_slice_2.update(value=(1, data_manager.get_size(selected_dataset_2)))

    metadata_type_2 = data_manager.get_metadata_type(selected_dataset_2)
    metadata_domain_2 = data_manager.get_metadata_domain(selected_dataset_2)

    while len(metadata_filters_2) > 0:
        metadata_filters_2.pop()
    for attribute in metadata_type_2:
        m_type = metadata_type_2[attribute]
        m_domain = metadata_domain_2[attribute]
        if m_type == 'boolean':
            filter = Select(title=attribute, value="Any",
                            options=["Any", "True", "False"])
        elif m_type == 'numerical':
            filter = RangeSlider(start=m_domain[0], end=m_domain[1],
                                 value=m_domain, step=1, title=attribute)
        elif m_type == 'categorical':
            categories = sorted(list(metadata_domain_2[attribute]))
            filter = MultiSelect(title=attribute, value=categories,
                                 options=categories)
        elif m_type == 'set':
            categories = sorted(list(metadata_domain_2[attribute]))
            filter = MultiSelect(title=attribute, value=categories,
                                 options=categories)
        else:
            raise ValueError(
                'Unsupported attribute type {} in metadata'.format(m_type))
        metadata_filters_2.append(filter)

    for df_group in data_filters_groups:
        df_group[4].update(
            options=['Both', selected_dataset_1, selected_dataset_2])
        if df_group[4].value not in ['Both', selected_dataset_1,
                                     selected_dataset_2]:
            df_group[4].update(value='Both')

    for control in metadata_filters_1 + metadata_filters_2:
        if hasattr(control, 'value'):
            control.on_change('value', update)
        if hasattr(control, 'active'):
            control.on_change('active', update)

    inputs.children = build_controls()

    update(attr, old, new)


def update_viz(attr, old, new):
    if opacity.value != scatter_1.glyph.fill_alpha:
        scatter_1.glyph.fill_alpha = opacity.value
        scatter_2.glyph.fill_alpha = opacity.value
        segments.line_alpha = opacity.value

    new_axes_font_size = str(int(axes_font_size.value)) + 'pt'
    plot.title.text_font_size = new_axes_font_size
    plot.xaxis.axis_label_text_font_size = new_axes_font_size
    plot.xaxis.major_label_text_font_size = new_axes_font_size
    plot.yaxis.axis_label_text_font_size = new_axes_font_size
    plot.yaxis.major_label_text_font_size = new_axes_font_size

    if args.labels:
        labels_annotations_1.text_alpha = int(show_labels.value == 'True')
        labels_annotations_1.text_font_size = str(
            int(labels_font_size.value)) + 'pt'
        labels_annotations_2.text_alpha = int(show_labels.value == 'True')
        labels_annotations_2.text_font_size = str(
            int(labels_font_size.value)) + 'pt'


def add_data_filter():
    data_filters.append(Div(text="<hr/>"))
    data_filter = build_data_filter()
    data_filters.extend(data_filter)
    data_filters_groups.append(data_filter)
    inputs.children = build_controls()


def build_controls():
    controls = [dataset_1, dataset_2, measure, x_axis, y_axis, items,
                visualization_title, opacity, axes_font_size,
                metadata_filters_title_1, rank_slice_1, *metadata_filters_1,
                metadata_filters_title_2, rank_slice_2, *metadata_filters_2,
                data_filters_title, difference_slider, slope_slider,
                *data_filters]

    dataset_1.on_change('value', update_dataset)
    dataset_2.on_change('value', update_dataset)
    measure.on_change('value', update)
    x_axis.on_change('value', update)
    y_axis.on_change('value', update)
    items.on_change('value', update)
    opacity.on_change('value', update_viz)
    axes_font_size.on_change('value', update_viz)
    if args.labels:
        controls.insert(9, show_labels)
        show_labels.on_change('value', update_viz)
        controls.insert(10, labels_font_size)
        labels_font_size.on_change('value', update_viz)
    rank_slice_1.on_change('value', update)
    rank_slice_2.on_change('value', update)
    difference_slider.on_change('value', update)
    slope_slider.on_change('value', update)
    controls.append(add_data_filter_button)
    if not add_data_filter_button._callbacks.get('clicks'):
        add_data_filter_button.on_click(add_data_filter)

    for control in metadata_filters_1 + metadata_filters_2 + data_filters:
        if hasattr(control, 'value'):
            control.on_change('value', update)
        if hasattr(control, 'active'):
            control.on_change('active', update)

    return controls


inputs = widgetbox(build_controls(), sizing_mode='fixed')
l = layout(
    [
        [plot, inputs],
    ]
    , sizing_mode='fixed')

update(None, None, None)  # initial load of the data

curdoc().add_root(l)
curdoc().title = "Comparison"
