
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


## Download dataset

First, using the code below, we upload the kaggle.json file to the colab runtime. Next, we move this file to the location from which we want to read it using the API.

```python
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

## Extract data

First, let's extract the entire contents of the ZIP file to the base location of the session. After extracting, let's delete the unnecessary file to reduce the amount of memory space used.

```python
path = '/content/covid19-radiography-database.zip'
with zipfile.ZipFile(path, 'r') as zip:
  zip.extractall()
os.remove(path)
```

Let's search all folders containing images for image file names. Let's save these names in a separate list and extract them to get a vector.
```python
filenames = list()
dataset_path = '/content/COVID-19_Radiography_Dataset'
for dir, _, filename in os.walk(dataset_path):
  if 'images' in dir:
    filenames.append(filename)
filenames = [item for sublist in filenames for item in sublist]
```

Additionally, let's create a dictionary containing the full path for each file name.

```python
all_image_paths = {os.path.basename(x): x for x in glob.glob(os.path.join(dataset_path, '*/', 'images', '*.png'))}
```

Using the pandas library, let's build a new dataframe object with the first column being the previously created list of file names. Then, starting from the top, we create new columns in which one is the full path of the file and the other is the extracted class name. We remove the 'image' column, which is redundant at this point. In the last steps, we shuffle the dataset and reset its index.

```python
df = pd.DataFrame(data = {'image': filenames})
df['path'] = df.image.map(all_image_paths)
df['label'] = df.image.apply(lambda x: x.split('-')[0])
df.drop(columns = 'image', inplace = True)
df = df.sample(frac = 1)
df.reset_index(drop = True, inplace = True)
```


## Data preprocessing

Change the class names of the dataset with the LabelEncoder from strings to int values. This will give us an array of encoded values that we can convert further.

```python
encoder = LabelEncoder()
labels = encoder.fit_transform(df.values[:,1])
```

Let's balance the distribution of classes in the dataset using the calculation of weights for each sample. 

```python
class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                  classes = np.unique(labels),
                                                  y = labels)
weights_dict = {np.unique(labels)[i]: class_weights[i] for i in range(len(class_weights))}
weights = np.asarray(list(map(weights_dict.get, labels)))
```

Additionally, let's hot encode these values.

```python
hot_encoder = OneHotEncoder()
labels = hot_encoder.fit_transform(df.values[:,1].reshape(-1, 1))
labels = labels.toarray()
```

Each of the previously created arrays is converted into tensors and then merged into the input pipline dataset object.

```python
path_tensor = tf.convert_to_tensor(df.values[:,0])
label_tensor = tf.convert_to_tensor(labels)
weight_tensor = tf.convert_to_tensor(weights)
dataset = tf.data.Dataset.from_tensor_slices(tensors = (path_tensor, label_tensor, weight_tensor))
```

Each record in a dataset object now consists of 3 elements - image path, hot encoded label and corresponding weight. To get the matching input for our model we need to perform some additional operations. For this purpose we will use the custom function load_images, which loads an image from the full file path and returns all the elements in the record in the appropriate format. We apply the function to the dataset object using map().

```python
def load_images(path, label, weight, IMG_SIZE = (224, 224)):
  image = tf.io.read_file(path)
  image = tf.image.decode_image(image, channels = 3,
                                dtype = tf.float16,
                                expand_animations = False)
  image = tf.image.resize(image, IMG_SIZE)
  label = tf.cast(label, dtype = tf.float16)
  weight = tf.cast(weight, dtype = tf.float16)
  return image, label, weight

dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
```

The resulting output object is presented as follows.

```python
<ParallelMapDataset element_spec=(TensorSpec(shape=(224, 224, 3), dtype=tf.float32, name=None), TensorSpec(shape=(4,), dtype=tf.float16, name=None), TensorSpec(shape=(), dtype=tf.float16, name=None))>
```

The next step will be to divide the collection into 3 parts in the proportion of 70% - 15% - 15%, constituting in turn the training collection, the validation collection and the test collection. To do this we first need to read the total number of available samples.

```python
DATASET_SIZE = len(dataset)

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)
```

The final element will be to divide the set into groups - batches, in this case they will be 64 elements. Batches characterize how many images our model will process simultaneously. It will also increase the dimensionality. Additionally, the elements of the dataset are prefetched, it means that the later elements are prefetched when the current element is processed.

```python
train_dataset = train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
```

Let's now examine one of the split datasets, let's take this training example. You can see that an additional fourth dimension, referred to as batch_size, has appeared in the element describing the image input to the model.

```
<PrefetchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 4), dtype=tf.float16, name=None), TensorSpec(shape=(None,), dtype=tf.float16, name=None))>
```
