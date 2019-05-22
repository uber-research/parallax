Parallax
========

Parallax is a tool for visualizing embeddings. It allows you to visualize the embedding space selecting explicitly the axis through algebraic formulas on the embeddings (like `king-man+woman`) and highlight specific items in the embedding space.
It also supports implicit axes via PCA and t-SNE.
There are three main views: the cartesian view that enables comparison on two user defined dimensions of variability (defined through formulae on embeddings), the comparison view that is similar to the cartesian but plots points from two datasets at the same time, and the polar view, where the user can define multiple dimensions of variability and show how a certain number of items compare on those dimensions.
For a thorough description please read the paper: https://www.overleaf.com/read/dgkgzkrcbdcq

## Set Up Environment (using virtualenv is not required)
```
virtualenv -p python3 venv
. venv/bin/activate
pip install -r requirements.txt
```

## Download example data
In order to replicate the experiments in our paper, you can download the files in:
```
https://drive.google.com/open?id=19EYhNu1Q-tRMKrGd05VF9nRVVbKEM-u3
```
and place the four files in `data` folder.
The Google Drive folder contains the Gigaword+Wikipedia and Twitter embeddings trained with [GloVe](https://nlp.stanford.edu/projects/glove/), plus some metadata we obtained by the automatic script in `modules/generate_metadata.py`.


## Run
```
bokeh serve --show cartesian.py
```
```
bokeh serve --show comparison.py
```
```
bokeh serve --show polar.py
```
You can add additional arguments like this:
```
bokeh serve --show cartesian.py --args -k 20000 -l -d '...'
```
- `-d` or `--datasets` loads custom embeddings. It accepts a JSON string containing a list of dictionaries. Each dictionary should contain a name field, an embedding_file field and a metadata_file field.  For example: `[{"name": "wikipedia", "embedding_file": "...", "metadata_file": "..."}, {"name": "twitter", "embedding_file": "...", "metadata_file": "..."}]` As it is a JSON string passed as a parameter, do not forget to escape the double quotes:
```
bokeh serve --show cartesian.py --args "[{\"name\": \"wikipedia\", \"embedding_file\": \"...\", \"metadata_file\": \"...\"}, {\"name\": \"twitter\", \"embedding_file\": \"...\", \"metadata_file\": \"...\"}]""
```
- `-k` or `--first_k` loads only the first `k` embeddings from the embeddings files. This assumes that the embedding in those files are sorted by unigram frequency in the dataset used for learning the embeddings (that is true for the pretrained GloVe embeddings for instance) so you are loading the k most frequent ones.
- `-l` or `--lables` gives you the option to show the level of the embedding in the scatterplot rather than relying on the mousehover. Because of the way bokeh renders those labels, this makes the scatterplot much slower, so I suggest to use it with no more than 10000 embeddings.
The comparison view requires at least two datasets to load.

## Custom Datasets
If you want to use your own data, the format of the embedding file should be like the GloVe one:
```
label1 value1_1 value1_2 ... value1_n
label2 value2_1 value2_2 ... value2_n
...
```
while the metadata file is a json file that looks like this:
```
{
  "types": {
    "length": "numerical",
    "pos tag": "set",
    "stopword": "boolean"
  },
  "values": {
    "overtones": {"length": 9, "pos tag": ["Noun"], "stopword": false},
    "grizzly": {"length": 7, "pos tag": ["Adjective Sat", "Noun"], "stopword": false},
  }
}
```
You can define your own type names, the supported types are boolean, numerical, categorical and set.
Each key in the values dictionary is one label in the embeddings file and the associated dict has one key for each type name in the types dictionary and the actual value for that specific label.
More in general:
```
{
  "types": {
    "type_name_1": ["numerical" | "binary" | categorical" | "set"],
    "type_name_2": ["numerical" | "binary" | categorical" | "set"],
    ...
  },
  "values": {
    "label_1": {"type_name_1": value, "type_name_2": value, ...},
    "label_2": {"type_name_1": value, "type_name_2": value, ...},
  }
}
```

#### Little caveat
Formulae are evaluated as pieces of python code through the `eval()` function.
Because of that, there are some limitations regarding the labels allowed for the embeddings (the same rules that apply for python3 variable naming):
- Variables names must start with a letter or an underscore, such as:
   - _underscore
   - underscore_
- The remainder of your variable name may consist of letters, numbers and underscores.
   - password1
   - n00b
   - un_der_scores
- Names are case sensitive.
   - case_sensitive, CASE_SENSITIVE, and Case_Sensitive are each a different variable.
- Variable names must not be one of python protected keywords
   - and, as, assert, break, class, continue, def, del, elif, else, except, exec, finally, for, from, global, if, import, in, is, lambda, not, or, pass, print, raise, return, try, while, with, yield
Lables that don't respect those rules will simply not be resolved in the formulae.
There could be solutions to this, we are already working on one.

## To Do
- display errors in the UI rather than in the console prints (there's no simple way in bokeh to do it)
- add clustering of points
- solve the issue that embedding labels have to conform to python variable naming conventions.

## Known Issues
- t-SNE is slow to compute, should require a loading UI.
- the scatterplot is in webgl and it's pretty fast even with hundreds of thousands of datapoints. With the labels enabled it uses html canvas that is really slow, so you may want to reduce the number of embeddings to less than 10000 for a responsive UI.
- In the polar view, changing axes and items can result in wrong legends
- The polar view rendering strategy is clearly sub-optimal

## Parallax v2
We are already working on a v2 of parallax that will not be using Bokeh but deck.gl for much improved performance with webgl rendering of labels, much faster implementation of t-SNE and more streamlined process for loading data. Stay tuned and reach out if you want to contribute!
