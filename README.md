# id-cuisi: Identifying Cultural Cuisine Influences in Recipes
## What is this?
This is the corpus of code and data that I am currently developing and exploring (respectively) as part of my final project for the Fall 2016 Machine Learning class taught by [Prof. Sandoval](http://engineering.nyu.edu/people/gustavo-sandoval) at [NYU's Tandon School of Engineering](http://engineering.nyu.edu/).

I'm investigating methods of classifying recipes based on ingredient and instruction data. Ultimately, I want to develop something which can identify not just a primary cultural influence but potentially secondary or tertiary influences as well.

While I may need to add data in the future, currently all data is part of the ["What's Cooking?"](https://www.kaggle.com/c/whats-cooking/data) data set available on [Kaggle](https://www.kaggle.com/) (see link below). This dataset was provided by Yummly as part of a competition on Kaggle, encouraging users to develop techniques by which to classify recipes using their ingredient lists. This dataset consists 39,774 distinct recipes in JSON format, where each recipe is represented by an ID, a cuisine type and a list of ingredients (measurements have been scrubbed).

## How do I use this?
There's only one functional script at present ([baseline.py](https://github.com/cmcg513/id-cuisi/blob/master/baseline.py)) and all it does right now is generate a model, test that model against train and test data, and then report some metrics on its performance. Additional scripts and functionality will be added over time. That said, if you wish to play around with it, by all means!

To run it you'll need [Python](https://www.python.org/) 3. Any package dependencies can be resolved using the [requirements.txt](https://github.com/cmcg513/id-cuisi/blob/master/requirements.txt) file. Note that I just did a pip freeze to generate that file, so there are likely packages in there that aren't actually dependencies for this project (I'll fix this in a future commit).

Below is the help text about how to run the script. It's fairly straighforward:
```
usage: baseline.py [-h] [-f FILEPATH]

baseline.py: uses a straightforward Naive Bayes classification method on
recipe data and reports some statistics/metrics on its performance

optional arguments:
  -h, --help            show this help message and exit
  -f FILEPATH, --file FILEPATH
                        the filepath to the dataset; defaults to relative
                        path: data/train.json
```
