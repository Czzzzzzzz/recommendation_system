# Contents

## [Datasets](datasets)

Datasets module includes helper functions for pulling different datasets and formatting them appropriately as well as utilities for splitting data for training / testing.

### Data Loading

There are dataloaders for several datasets. For example, the movielens module will allow you to load a dataframe in pandas or spark formats from the MovieLens dataset, with sizes of 100k, 1M, 10M, or 20M to test algorithms and evaluate performance benchmarks.

```python
df = movielens.load_pandas_df(size="100k")
```

### Splitting Techniques

Currently three methods are available for splitting datasets. All of them support splitting by user or item and filtering out minimal samples (for instance users that have not rated enough items, or items that have not been rated by enough users).

- Random: this is the basic approach where entries are randomly assigned to each group based on the ratio desired
- Chronological: this uses provided timestamps to order the data and selects a cut-off time that will split the desired ratio of data to train before that time and test after that time
- Stratified: this is similar to random sampling, but the splits are stratified, for example if the datasets are split by user, the splitting approach will attempt to maintain the same ratio of items used in both training and test splits. The converse is true if splitting by item.

## [Evaluation](evaluation)

The evaluation submodule includes functionality for calculating common recommendation metrics directly in Python or in a Spark environment using PySpark.

Currently available metrics include:

- Root Mean Squared Error
- Mean Absolute Error
- R<sup>2</sup>
- Explained Variance
- Precision at K
- Recall at K
- Normalized Discounted Cumulative Gain at K
- Mean Average Precision at K
- Area Under Curve
- Logistic Loss

## [Models](models)

The models submodule contains implementations of various algorithms that can be used in addition to external packages to evaluate and develop new recommender system approaches. The following is a list of the algorithm utilities:

## [Tuning](tuning)

This submodule contains utilities for performing hyperparameter tuning.

## [Utils](utils)

This submodule contains high-level utilities for defining constants used in most algorithms as well as helper functions for managing aspects of different frameworks: GPU, Spark, Jupyter notebook.
