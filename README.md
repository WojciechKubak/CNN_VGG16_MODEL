
## Required modules

Download and extraction
```python
from google.colab import files
import numpy as np
import zipfile
import os
import glob
```

Analysis and visualization
```python
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
import cv2
```

Feature engineering
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
```

Model building and evaluation
```python
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
```

## Characteristics of the dataset

The analyzed collection represents more than 30,000 images divided according to the four available classes as follows:

![alt text](https://github.com/WojciechKubak/CNN_VGG16_XRAY/blob/main/Images/class_distribution.png?raw=true)

There is a high prevalence of some class samples in the dataset. In order for the model to learn correctly, it will be necessary to perform some operations to balance the distribution of classes in the dataset in question. For example, by giving correlating weights or using only a part of the whole set. 

The images are saved in PNG format, in folders corresponding to the class. Each image has a size of 256 x 256 pixels. Below are example samples for each of the classes available in the collection:

![alt text](https://github.com/WojciechKubak/CNN_VGG16_XRAY/blob/main/Images/random_samples.png?raw=true)


## Extracting data

First, using the code below, we upload the kaggle.json file to the colab runtime. Next, we move this file to the location from which we want to read it using the API.

```python
from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

From here, we can already download the dataset of interest from the kaggle platform using the command:

```python
!kaggle datasets download -d tawsifurrahman/covid19-radiography-database
```

The file will be downloaded in ZIP format to the base location.
