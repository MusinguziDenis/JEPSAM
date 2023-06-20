# Realization of the Situation model Framework (SMF) using a Joint Episodic-Procedural-Semantic Associative Memory (JEPSAM)

This repository includes the implementation of the JEPSAM as an autoencoder (AEJEPS)


## I- Contents 
1. [Configs](./configs/)

> This folder contains the config files that define the hyperparameters and other variables useful to run the training and evaluation scripts.

2. [src](./src/)

> This folder contains all the training and evaluation scripts as well as the utility funtions.

## II- Usage
All scripts should be executed from the parent folder `AEJEPS`.
1. Data loaders

The Data loader is instanciated using the `AllModalitiesDataset` class. Below is an example:

```python
train_ds                = AllModalitiesDataset(
    csv                 = csv_path,
    dataset_directory   = dataset_path
)

train_dl                = torch.utils.data.DataLoader(
    dataset             = train_ds,
    batch_size          = 16,
    shuffle             = True,
    num_workers         = 2,
    pin_memory          = True,
    collate_fn          = train_ds.collate_fn
)
```

You can test the data loading pipeline as follows:

```bash
(jeps)$ python src/dataloaderV1.py
```
Below is an example of the expected output:
```bash
INFO:root:Checking data loading pipeline
INFO:root:Prepared 1489 training samples and 483 validation samples 
INFO:root:
>> train data loader
In		    : torch.Size([16, 1000, 1000, 3])
Goal		: torch.Size([16, 1000, 1000, 3])
Action desc	: torch.Size([8, 16])
CMD		    : torch.Size([9, 16])

INFO:root:
>> val data loader
In		    : torch.Size([16, 1000, 1000, 3])
Goal		: torch.Size([16, 1000, 1000, 3])
Action desc	: torch.Size([8, 16])
CMD		    : torch.Size([9, 16])

```

## III- Acknowledgements

TBA

## IV- Citations

TBA



