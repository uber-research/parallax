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

from bokeh.io import curdoc
from bokeh.layouts import layout, column, widgetbox
from bokeh.models import ColumnDataSource, HoverTool, Div, LabelSet, \
    RangeSlider, MultiSelect, Line, Panel, \
    RadioButtonGroup, Tabs, Button
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
                         'For example: \'[{"name": "wikipedia", "embeddings_file": "...", "metadata_file": "..."}, '
                         '{"name": "twitter", "embeddings_file": "...", "metadata_file": "..."}]\'')
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

default_color = 'steelblue'
highlight_color = 'tomato'
last_embeddings = None

data_manager = DataManager(args.datasets, args.first_k)
selected_dataset = data_manager.dataset_ids[0]

# Create Column Data Source that will be used by the plot
source = ColumnDataSource(data=dict(x=[], y=[], color=[], label=[]))
source_items = ColumnDataSource(data=dict(x=[], y=[], color=[], label=[]))
line_source = ColumnDataSource(dict(x=[], y=[]))

hover = HoverTool(tooltips=[
    ("Label", "@label"),
    ("x", "@x"),
    ("y", "@y")
],
    names=["scatter"])

plot = figure(plot_height=800, plot_width=800, title="",
              toolbar_location='right', output_backend=args.output_backend)
plot.add_tools(hover)
line = Line(x='x', y='y', line_width=2)
plot.add_glyph(line_source, line)
scatter = plot.circle(x="x", y="y", source=source, size=7, color="color",
                      line_color=None, fill_alpha=0.6, name='scatter')
if args.labels:
    labels_annotations = LabelSet(x="x", y="y", text="label", y_offset=1,
                                  text_font_size="8pt", text_color="#555555",
                                  source=source, text_align='center',
                                  text_alpha=0)
    plot.add_layout(labels_annotations)
    labels_annotations_items = LabelSet(x="x", y="y", text="label", y_offset=1,
                                        text_font_size="8pt",
                                        text_color="#555555",
                                        source=source_items,
                                        text_align='center', text_alpha=0)
    plot.add_layout(labels_annotations_items)

# Create Input controls
dataset = Select(title="Dataset", options=data_manager.dataset_ids,
                 value=selected_dataset)

measure_1 = Select(title="Measure",
                   options=[('cosine_similarity', 'Cosine Similarity'),
                            ('cosine', 'Cosine Distance'),
                            ('euclidean', 'Euclidean'),
                            ('dot_product', 'Dot product'),
                            ('correlation', 'Correlation'),
                            ],
                   value='cosine_similarity')
x_axis = TextInput(title="X Axis Formula", placeholder="formula")
y_axis = TextInput(title="Y Axis Formula", placeholder="formula")
explicit_tab = Panel(child=column(measure_1, x_axis, y_axis), title="Explicit")

filtering_before_after_2 = RadioButtonGroup(
    labels=["Filter before projection", "Filter after projection"], active=0)
pca_tab = Panel(child=filtering_before_after_2, title="PCA")

measure_3 = Select(title="Measure",
                   options=[('cosine', 'Cosine Distance'),
                            ('euclidean', 'Euclidean'),
                            ('dot_product', 'Dot product'),
                            ('correlation', 'Correlation'),
                            ],
                   value='cosine')
filtering_before_after_3 = RadioButtonGroup(
    labels=["Filter before projection", "Filter after projection"], active=0)
perplexity = TextInput(title="Perplexity", value='30')
early_exaggeration = TextInput(title="Early exaggeration", value='12.0')
learning_rate = TextInput(title="Learning rate", value='200.0')
n_iter = TextInput(title="# Iterations", value='1000')
n_iter_without_progress = TextInput(title="# Iterations without progress",
                                  value='300')
min_grad_norm = TextInput(title="Min grad norm", value='1e-7')
init = Select(title="Init",
              options=['pca', 'random'],
              value='pca')
method = Select(title="Method",
                options=[('barnes_hut', 'Barnes Hut'),
                         ('exact', 'Exact')],
                value='barnes_hut')
angle = TextInput(title="Angle", value='0.5')
tsne_tab = Panel(child=column(measure_3, filtering_before_after_3, perplexity,
                              early_exaggeration, learning_rate, n_iter,
                              n_iter_without_progress, min_grad_norm, init,
                              method, angle),
                 title="t-SNE")

projection_tab_panel = Tabs(tabs=[explicit_tab, pca_tab, tsne_tab])

items = TextInput(title='Items Formulae (separated by ";")',
                  placeholder="formula; formula; ...")

filters_title = Div(text="<strong>Metadata Filters</strong>")

rank_slice = RangeSlider(title="Rank Slice",
                         value=(1, data_manager.get_size(selected_dataset)),
                         start=1,
                         end=data_manager.get_size(selected_dataset), step=1)

metadata_type = data_manager.get_metadata_type(selected_dataset)
metadata_domain = data_manager.get_metadata_domain(selected_dataset)

metadata_filters = []
for attribute in metadata_type:
    m_type = metadata_type[attribute]
    m_domain = metadata_domain[attribute]
    if m_type == 'boolean':
        filter = Select(title=attribute, value="Any",
                        options=["Any", "True", "False"])
    elif m_type == 'numerical':
        filter = RangeSlider(start=m_domain[0], end=m_domain[1],
                             value=m_domain, step=1, title=attribute)
    elif m_type == 'categorical':
        categories = sorted(list(metadata_domain[attribute]))
        filter = MultiSelect(title=attribute,  # value=categories,
                             options=categories)
    elif m_type == 'set':
        categories = sorted(list(metadata_domain[attribute]))
        filter = MultiSelect(title=attribute,  # value=categories,
                             options=categories)
    else:
        raise ValueError(
            'Unsupported attribute type {} in metadata'.format(m_type))
    metadata_filters.append(filter)
filters_column = column(*metadata_filters)

data_filters_title = Div(text="<strong>Data Filters</strong>")


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
                                 ('close', '~'),
                                 ('less_equal', '≤'),
                                 ('less', '<'),
                                 ],
                        value='greater', width=20)
    value_cf = TextInput(value='', placeholder="numeric value")
    data_filter = [measure_cf, formula_cf, compare_cf, value_cf]
    return data_filter


data_filter = build_data_filter()
data_filters = [widget for widget in data_filter]
data_filters_groups = [data_filter]
add_data_filter_button = Button(label="Add", button_type="success")

visualization_title = Div(text="<strong>Visualization</strong>")

opacity = Slider(title="Opacity", value=0.6, start=0, end=1, step=0.01)
axes_font_size = Slider(title="Axes Font Size", value=8, start=8, end=32)
if args.labels:
    show_labels = RadioButtonGroup(
        labels=["No labels", "Item labels", "All labels"],
        active=0)
    labels_font_size = Slider(title="Labels Font Size", value=8, start=8,
                              end=32)
    labels_opacity = Slider(title="Lales Opacity", value=1, start=0, end=1,
                            step=0.01)
    labels_items_opacity = Slider(title="Item Labels Opacity", value=1, start=0,
                                  end=1, step=0.01)


def select_embeddings():
    x_axis_value = x_axis.value.strip()
    y_axis_value = y_axis.value.strip()
    if projection_tab_panel.active == 0 and (
            x_axis_value == '' or y_axis_value == ''):
        return {}
    else:
        metadata_filters_params = []
        for metadata_filter in metadata_filters:
            if metadata_type[metadata_filter.title] == 'boolean':
                if metadata_filter.value != 'Any':
                    metadata_filters_params.append((metadata_filter.title,
                                                    metadata_filter.value == 'True'))
            elif metadata_type[metadata_filter.title] == 'numerical':
                filter_value = (
                    int(metadata_filter.value[0]),
                    int(metadata_filter.value[1]))
                if filter_value != (int(rank_slice.start), int(rank_slice.end)):
                    metadata_filters_params.append(
                        (metadata_filter.title, filter_value))
            elif metadata_type[metadata_filter.title] == 'categorical':
                filter_value = set(metadata_filter.value)
                if len(filter_value) > 0 and filter_value != metadata_domain[
                    metadata_filter.title]:
                    metadata_filters_params.append(
                        (metadata_filter.title, filter_value))
            elif metadata_type[metadata_filter.title] == 'set':
                filter_value = set(metadata_filter.value)
                if len(filter_value) > 0 and filter_value != metadata_domain[
                    metadata_filter.title]:
                    metadata_filters_params.append(
                        (metadata_filter.title, filter_value))

        data_filters_params = []
        for data_filter in data_filters_groups:
            formula = data_filter[1].value.strip()
            number_value = data_filter[3].value.strip()
            if formula != '' and number_value != '':
                measure = data_filter[0].value
                compare_function = data_filter[2].value
                try:
                    number = float(number_value)
                    data_filters_params.append({'measure': measure,
                                                'formula': formula,
                                                'compare_function': compare_function,
                                                'number': number})
                except:
                    print('invalid number value:', number_value)

        items_list = []
        items_val = items.value.strip()
        if items_val != '':
            items_list = re.split("\s*;\s*", items_val)

        additional_arguments = {}
        if projection_tab_panel.active == 0:  # explicit
            mode = 'explicit'
            metric = measure_1.value
            formulae = [x_axis_value, y_axis_value]
            pre_filtering = True
            post_filtering = False
        elif projection_tab_panel.active == 1:  # pca
            mode = 'pca'
            metric = None
            formulae = None
            pre_filtering = filtering_before_after_2.active == 0
            post_filtering = filtering_before_after_2.active == 1
        else:  # if projection_tab_panel.active == 2:  # tsne
            mode = 'tsne'
            metric = measure_3.value
            formulae = None
            pre_filtering = filtering_before_after_3.active == 0
            post_filtering = filtering_before_after_3.active == 1

            additional_arguments['perplexity'] = float(perplexity.value)
            additional_arguments['early_exaggeration'] = float(early_exaggeration.value)
            additional_arguments['learning_rate'] = float(learning_rate.value)
            additional_arguments['n_iter'] = int(n_iter.value)
            additional_arguments['n_iter_without_progress'] = int(n_iter_without_progress.value)
            additional_arguments['min_grad_norm'] = float(min_grad_norm.value)
            additional_arguments['init'] = init.value
            additional_arguments['method'] = method.value
            additional_arguments['angle'] = float(angle.value)

        rank_slice_values = (int(rank_slice.value[0]), int(rank_slice.value[1]))
        if rank_slice_values == (int(rank_slice.start), int(rank_slice.end)):
            rank_slice_values = None

        return projection(data_manager,
                          dataset_id=dataset.value,
                          data_filters=data_filters_params,
                          metadata_filters=metadata_filters_params,
                          mode=mode,
                          rank_slice=rank_slice_values,
                          metric=metric,
                          n_axes=2,
                          formulae=formulae,
                          items=items_list,
                          pre_filtering=pre_filtering,
                          post_filtering=post_filtering,
                          **additional_arguments
                          )


def update_view(embeddings):
    if projection_tab_panel.active == 0:  # explicit
        plot.xaxis.axis_label = x_axis.value
        plot.yaxis.axis_label = y_axis.value
        line.line_alpha = 1
    else:
        plot.xaxis.axis_label = ''
        plot.yaxis.axis_label = ''
        line.line_alpha = 0

    if projection_tab_panel.active == 0:  # explicit
        measure_name = ' '.join(measure_1.value.split('_'))
        plot.title.text = "{} embeddings selected, {} measure".format(
            len(embeddings), measure_name)
    elif projection_tab_panel.active == 2:  # t-SNE
        measure_name = ' '.join(measure_3.value.split('_'))
        plot.title.text = "{} embeddings selected, {} measure".format(
            len(embeddings), measure_name)
    else:
        plot.title.text = "{} embeddings selected".format(len(embeddings))

    items_list = []
    items_val = items.value.strip()
    if items_val != '':
        items_list = re.split("\s*;\s*", items.value.strip())
    highlight_items = set(items_list)

    x = []
    y = []
    labels = []
    colors = []
    x_item = []
    y_item = []
    labels_item = []
    colors_item = []
    for label, emd_dict in embeddings.items():
        x.append(emd_dict['coords'][0])
        y.append(emd_dict['coords'][1])
        labels.append(label)
        colors.append(
            highlight_color if label in highlight_items else default_color)

        if label in highlight_items:
            x_item.append(emd_dict['coords'][0])
            y_item.append(emd_dict['coords'][1])
            labels_item.append(label)
            colors_item.append(
                highlight_color if label in highlight_items else default_color)

    source.data = dict(
        x=x,
        y=y,
        color=colors,
        label=labels,
    )

    source_items.data = dict(
        x=x_item,
        y=y_item,
        color=colors_item,
        label=labels_item,
    )

    if len(x) > 0 and len(y) > 0:
        line_min = min(min(x), min(y))
        line_max = max(max(x), max(y))
        line_source.data = dict(x=[line_min, line_max], y=[line_min, line_max])


def update(attr, old, new):
    global last_embeddings
    embeddings = select_embeddings()
    last_embeddings = embeddings
    update_view(embeddings)


def update_items(attr, old, new):
    global last_embeddings
    embeddings = last_embeddings if last_embeddings is not None else select_embeddings()
    update_view(embeddings)


def update_dataset(attr, old, new):
    selected_dataset = dataset.value
    rank_slice.end = data_manager.get_size(selected_dataset)
    rank_slice.update(value=(1, data_manager.get_size(selected_dataset)))

    metadata_type = data_manager.get_metadata_type(selected_dataset)
    metadata_domain = data_manager.get_metadata_domain(selected_dataset)

    while len(metadata_filters) > 0:
        metadata_filters.pop()
    for attribute in metadata_type:
        m_type = metadata_type[attribute]
        m_domain = metadata_domain[attribute]
        if m_type == 'boolean':
            filter = Select(title=attribute, value="Any",
                            options=["Any", "True", "False"])
        elif m_type == 'numerical':
            filter = RangeSlider(start=m_domain[0], end=m_domain[1],
                                 value=m_domain, step=1, title=attribute)
        elif m_type == 'categorical':
            categories = sorted(list(metadata_domain[attribute]))
            filter = MultiSelect(title=attribute, value=categories,
                                 options=categories)
        elif m_type == 'set':
            categories = sorted(list(metadata_domain[attribute]))
            filter = MultiSelect(title=attribute, value=categories,
                                 options=categories)
        else:
            raise ValueError(
                'Unsupported attribute type {} in metadata'.format(m_type))
        metadata_filters.append(filter)

    for control in metadata_filters:
        if hasattr(control, 'value'):
            control.on_change('value', update)
        if hasattr(control, 'active'):
            control.on_change('active', update)

    inputs.children = build_controls()

    update(attr, old, new)


def update_viz(attr, old, new):
    if opacity.value != scatter.glyph.fill_alpha:
        scatter.glyph.fill_alpha = opacity.value

    new_axes_font_size = str(int(axes_font_size.value)) + 'pt'
    plot.title.text_font_size = new_axes_font_size
    plot.xaxis.axis_label_text_font_size = new_axes_font_size
    plot.xaxis.major_label_text_font_size = new_axes_font_size
    plot.yaxis.axis_label_text_font_size = new_axes_font_size
    plot.yaxis.major_label_text_font_size = new_axes_font_size

    if args.labels:
        labels_annotations.text_alpha = int(
            show_labels.active == 2) * labels_opacity.value
        labels_annotations_items.text_alpha = int(
            show_labels.active == 1 or show_labels.active == 2) * labels_items_opacity.value
        labels_annotations.text_font_size = str(
            int(labels_font_size.value)) + 'pt'
        labels_annotations_items.text_font_size = str(
            int(labels_font_size.value)) + 'pt'


def add_data_filter():
    data_filters.append(Div(text="<hr/>"))
    data_filter = build_data_filter()
    data_filters.extend(data_filter)
    data_filters_groups.append(data_filter)
    inputs.children = build_controls()


def build_controls():
    controls = [dataset, projection_tab_panel, items,
                visualization_title, opacity, axes_font_size,
                filters_title, rank_slice, *metadata_filters,
                data_filters_title, *data_filters]

    dataset.on_change('value', update_dataset)
    measure_1.on_change('value', update)
    x_axis.on_change('value', update)
    y_axis.on_change('value', update)
    filtering_before_after_2.on_change('active', update)
    projection_tab_panel.on_change('active', update)
    items.on_change('value', update_items)
    opacity.on_change('value', update_viz)
    axes_font_size.on_change('value', update_viz)
    if args.labels:
        controls.insert(6, show_labels)
        show_labels.on_change('active', update_viz)
        controls.insert(7, labels_font_size)
        labels_font_size.on_change('value', update_viz)
        controls.insert(8, labels_opacity)
        labels_opacity.on_change('value', update_viz)
        controls.insert(9, labels_items_opacity)
        labels_items_opacity.on_change('value', update_viz)
    rank_slice.on_change('value', update)
    controls.append(add_data_filter_button)
    if not add_data_filter_button._callbacks.get('clicks'):
        add_data_filter_button.on_click(add_data_filter)

    for control in metadata_filters + data_filters:
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
curdoc().title = "Cartesian"
