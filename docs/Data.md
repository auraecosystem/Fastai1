# Data loading adapter

Fastai1's maintained data adapter is available from
model_training.lmlm_training.data. It is intentionally independent of the
historical fastai internal package and accepts ordinary PyTorch
Dataset and IterableDataset objects.

## Basic usage

~~~python
from model_training.lmlm_training.data import make_dataloaders

loaders = make_dataloaders(
    train_dataset,
    validation_dataset,
    batch_size=32,
    device="cpu",
)
for inputs, targets in loaders.train:
    ...
~~~

## Transform order

1. after_item: runs on each dataset item.
2. before_batch: runs on the list of items before collation.
3. PyTorch collate_fn: creates a batch.
4. The adapter moves tensor values to its configured device.
5. after_batch: runs on the batch.

Transforms can be one callable or a sequence of callables. Model-specific
tokenization, sample interpretation, and custom collation remain the
responsibility of the dataset/application. This module does not import or
reimplement Fastai's private dispatch, transform, notebook, or display APIs.

## Validation and test loaders

DataLoaders.from_datasets and make_dataloaders create training and validation
loaders. Training shuffles by default; validation does not.
DataLoaders.test_loader(dataset) derives settings and transforms from the
validation loader, disables shuffling, and keeps incomplete final batches.

The adapter is a generic iterable accepted by the native trainer. Pass
loaders.train and loaders.valid to train(..., validation_loader=...).

## Multiprocessing

When num_workers is greater than zero, datasets, collators, and transforms
must be picklable under the active multiprocessing start method. Prefer
top-level functions over lambdas for worker processes. Device transfer and
after_batch transforms run in the main training process.
