# Key Value Memory Networks by Keras

![Key Value Memory Networks](https://raw.githubusercontent.com/jojonki/key-value-memory-networks/images/kvmemnns.png)

See original thesis.
> Key-Value Memory Networks for Directly Reading Documents, Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, Jason Weston
https://arxiv.org/abs/1606.03126

There are still some bugs in my code, pull-requests will be appreciated! :beer:

## Setup

First of all, you need to download dataset.
```
$ ./download.sh
```



## Train

You can use presaved pickle files or build data dictionaries with `process_data.py`.

```
$ python train.py
```

## Evaluate
```
$ python evaluate.py -m saved_keras_model_path
```

## Interactive
```
$ python interactive.py -m saved_keras_model_path
```

