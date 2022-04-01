
## Required modules

Download and extraction:
```python
from google.colab import files
import zipfile
import os
import glob
import pandas as pd
```

Analysis and visualization:
```python
import plotly.express as px
import cv2
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
```

Data preprocessing:
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight
import numpy as np
import tensorflow as tf
```

Model building and evaluation:
```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Activation, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import numpy as np
```

## Characteristics of the dataset

The analyzed collection represents more than 30,000 images divided according to the 4 available classes as follows:

![alt text](https://github.com/WojciechKubak/CNN_VGG16_XRAY/blob/main/Images/class_distribution.png?raw=true)

Code used:

```python
fig = px.bar(data_frame = df,
             x = df.label.value_counts().index,
             y = df.label.value_counts().values,
             text_auto = True)

fig.update_layout(title = 'Class distribution', 
                  xaxis_title = "class name", 
                  yaxis_title = "number of occurrences", 
                  legend_title = "Legend Title")

fig.show()
```

There is a high prevalence of some class samples in the dataset. In order for the model to learn correctly, it will be necessary to perform some operations to balance the distribution of classes in the dataset in question. For example, by giving correlating weights or using only a part of the whole set. 

The images are saved in PNG format, in folders corresponding to the class. Each image has a size of 256 x 256 pixels. Below are example samples for each of the classes available in the collection:

![alt text](https://github.com/WojciechKubak/CNN_VGG16_XRAY/blob/main/Images/random_samples.png?raw=true)

Code used:

```python
fig = make_subplots(rows = 2,
                    cols = 4,
                    subplot_titles = df.label.unique(), 
                    vertical_spacing = 0.1)

for index, label in enumerate(df.label.unique()):
  label_mask = df.label == label
  
  sample = df.loc[label_mask, 'path'].sample(1)
  image = read_image(sample.values[0])
  fig.add_trace(go.Image(z = image), row = 1, col = index + 1)

  sample = df.loc[label_mask, 'path'].sample(1)
  image = read_image(sample.values[0])
  fig.add_trace(go.Image(z = image), row = 2, col = index + 1)

fig.update_xaxes(visible = False)
fig.update_yaxes(visible = False)

fig.show()
```

## Download dataset

First, using the code below, we upload the kaggle.json file to the colab runtime.

```python
uploaded = files.upload()
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
```

We move this file to the location from which we want to read it using the API.

```python
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
```

From here, we can already download the dataset of interest from the kaggle platform using the command.

```python
!kaggle datasets download -d tawsifurrahman/covid19-radiography-database
```

The file will be downloaded in ZIP format to the base location.

## Extract data

First, extract the entire contents of the ZIP file to the base location of the session. After extracting, let's delete the unnecessary file to reduce the amount of memory space used.

```python
path = '/content/covid19-radiography-database.zip'
with zipfile.ZipFile(path, 'r') as zip:
  zip.extractall()
os.remove(path)
```

Now search all folders containing images for image file names. Let's save these names in a separate list and extract them to get a vector.
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

Build a new **dataframe** object with the first column being the previously created list of file names. Then, starting from the top, we create new columns in which one is the full path of the file and the other is the extracted class name. We remove the 'image' column, which is redundant at this point. In the last steps, we shuffle the dataset and reset its index.

```python
df = pd.DataFrame(data = {'image': filenames})
df['path'] = df.image.map(all_image_paths)
df['label'] = df.image.apply(lambda x: x.split('-')[0])
df.drop(columns = 'image', inplace = True)
df = df.sample(frac = 1)
df.reset_index(drop = True, inplace = True)
```


## Data preprocessing

Change the class names of the dataset with the **LabelEncoder** from strings to int values. This will give us an array of encoded values that we can convert further.

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

Additionally, let's **hot encode** these values.

```python
hot_encoder = OneHotEncoder()
labels = hot_encoder.fit_transform(df.values[:,1].reshape(-1, 1))
labels = labels.toarray()
```

Each of the previously created arrays is converted into **tensors** and then merged into the input pipline dataset object.

```python
path_tensor = tf.convert_to_tensor(df.values[:,0])
label_tensor = tf.convert_to_tensor(labels)
weight_tensor = tf.convert_to_tensor(weights)
dataset = tf.data.Dataset.from_tensor_slices(tensors = (path_tensor, label_tensor, weight_tensor))
```

Each record in a dataset object now consists of 3 elements - image path, hot encoded label and corresponding weight. To get the matching input for model we need to perform some additional operations. For this purpose we will use the custom function load_images, which loads an image from the full file path and returns all the elements in the record in the appropriate format.

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
```

We apply the function to the dataset object using *map()*.

```python
dataset = dataset.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
```

The resulting output object is presented as follows.

```
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

The final element will be to divide the set into groups - **batches**, in this case they will be 64 elements. Batches characterize how many images model will process simultaneously. It will also increase the dimensionality. Additionally, the elements of the dataset are **prefetched**, it means that the later elements are prefetched when the current element is processed.

```python
train_dataset = train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
```

Let's now examine one of the split datasets, let's take this training example. You can see that an additional fourth dimension, referred to as *batch_size*, has appeared in the element describing the image input to the model.

```
<PrefetchDataset element_spec=(TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None, 4), dtype=tf.float16, name=None), TensorSpec(shape=(None,), dtype=tf.float16, name=None))>
```
## Building and training model

The **VGG16** architecture containing weights trained on the **ImageNet** collection was used to implement the project. The structure of the entire model is shown below.

![alt text](https://github.com/WojciechKubak/CNN_VGG16_XRAY/blob/main/Images/vgg16_architecture.png?raw=true)


Initialize the amount available from the class dataset and the input image size.

```python
NUM_CLASSES = 4
IMG_SIZE = (224, 224, 3)
```

To build the model a custom function *create_model* was created, as arguments it takes - number of available classes, image size and initialized learning rate.

Using the API, we load the model without the top layer, set the weights to untrainable and add a new input with the appropriate shape. Then, when the **feature extractor** is ready, we add a flatten layer and then 2 fully connected layers followed by the softmax on the output. 

The model is then compiled, **Adam** was used as the optimizer, loss is **categorical crossentropy** (because the labels are one hot encoded), and metrics is **accuracy**. The value of initialized learning rate was chosen based on tests on the model. 

```python
def create_model(classes, shape, INIT_LR = 3e-4):
  feature_extractor = VGG16(include_top = False, weights = 'imagenet')
  feature_extractor.trainable = False

  inputs = Input(shape = shape, dtype = tf.float16, name = 'input_layer')
  extractor = feature_extractor(inputs)

  flatten = Flatten(name = 'flatten_layer')(extractor)

  classifier = Dense(2048)(flatten)
  classifier = Activation('relu', dtype = tf.float32)(classifier)
  classifier = Dropout(0.4)(classifier)
  classifier = BatchNormalization()(classifier)

  classifier = Dense(classes)(classifier)
  outputs = Activation(activation = 'softmax', 
                       dtype = tf.float32, 
                       name = 'softmax_output')(classifier)

  model = Model(inputs = inputs, outputs = outputs)

  model.compile(optimizer = Adam(learning_rate = INIT_LR),
                loss = 'categorical_crossentropy', 
                metrics = 'accuracy')
  
  return model 
```

Create a model object from the previously created function.

```python
model = create_model(NUM_CLASSES, IMG_SIZE)
```

For a better and more efficient training of the model, we will use selected **callbacks** available in tensorflow. 

**EarlyStopping** allows the model to stop when ovefitting occurs. We set as parameters the metric to be monitored and patience, which specifies how many epochs the model may not improve before stopping.

```python
early_stopping = EarlyStopping(monitor = 'val_accuracy',
                              patience = 3,
                              verbose = 0,
                              restore_best_weights = False)
```

**LearningRateScheduler** allows the model to adjust the learning rate with each successive epoch, according to a defined scheduler.

```python
EPOCHS = 100
scheduler = lambda x: 1e-4 * 0.95 ** (x + EPOCHS)
lr_scheduler = LearningRateScheduler(schedule =scheduler, verbose = 0)
```

**ModelCheckpoint** tracks the indicated metric and records the weights for which its value is best. This preserves the learning progression of the model. 

```python
path = "CHECKPOINTS/cp.ckpt" 
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = path, 
                                                     montior = "val_accuracy",
                                                     save_best_only = True,
                                                     save_weights_only = True,
                                                     verbose = 0)
```

At this point, we can now train the created model using the *train_dataset* and simultaneously validate it using the *val_dataset*. We set the *verbose* value to 2, which gives us the current information about the model improvement with each epoch, we collect this information to the history object.

```python
history = model.fit(x=train_dataset,
                    epochs=EPOCHS,
                    validation_data=val_dataset, 
                    batch_size = BATCH_SIZE,
                    verbose=2,
                    callbacks=[early_stopping, lr_scheduler, model_checkpoint])
```

Now by plotting *.history* we can see how the error rate and accuracy for each epoch in the training and validation set evolved.

![alt text](https://github.com/WojciechKubak/CNN_VGG16_XRAY/blob/main/Images/model_history.png?raw=true)

By the values of the level formation of the loss function, it can be seen that **overfitting** does not occur. For both sets, it was possible to obtain a value exceeding 90% efficiency, which gives a satisfactory model result.
## Model evaluation

In order to **evaluate** model, another instance of it will be created, then we will load the best weights obtained by the ModelCheckpoint callback stored in the path. 

```python
_model = create_model(NUM_CLASSES, IMG_SIZE)
_model.load_weights(path)
```

Check performance of model with samples from the test set that have not been used before.

```python
loss, accuracy = _model.evaluate(test_dataset)
```

The results of the model evaluation are as follows:

```
50/50 [==============================] - 20s 165ms/step - loss: 0.2629 - accuracy: 0.8995
```

**Confusion matrix** will be used to validate the model. Confusion Matrix is an N×N matrix, where rows correspond to correct decision classes and columns correspond to decisions predicted by the classifier. The n-ij number at the intersection of row i and column j is the number of examples from the i-th class that were classified into the j-th class.

In order to build a confusion matrix, we must first perform several operations on the prediction data and the actual data. First we used predict() to get the predicted results for the previously used test set. Next, we will convert the values encoded as one hot, to an int type. We will perform this operation on the predictions and true values to unify the data.  

```python
y_pred = _model.predict(test_dataset)
y_pred = np.array([np.argmax(x) for x in y_pred])

y_true = [element[1] for element in test_dataset.unbatch().as_numpy_iterator()]
y_true = np.array([np.argmax(x) for x in y_true])
```

With all samples transformed, we can create a confusion matrix, an additional list containing the unique labels present in the dataset.  

```python
conf_matrix = tf.math.confusion_matrix(y_true, y_pred)
conf_matrix = conf_matrix.numpy()
class_names = encoder.classes_
```

In order to make the results more transparent, we will visualize the matrix as a graph.

![alt text](https://github.com/WojciechKubak/CNN_VGG16_XRAY/blob/main/Images/confusion_matrix.png?raw=true)

The values contained on the main diagonal of the matrix define the test set samples correctly classified by the trained model.

Used code:

```python
fig = px.imshow(conf_matrix,
                labels = dict(x = "predicted sample", y="true sample"),
                x = class_names,
                y = class_names,
                text_auto = True,
                title = 'Confusion matrix')

fig.update_traces(showlegend=False)
fig.update_xaxes(side="top")

fig.show()
```