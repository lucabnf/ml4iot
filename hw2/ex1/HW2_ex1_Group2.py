import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import zlib
import tensorflow_model_optimization as tfmot

def check_type(x):
    if (x != 'a') and (x != 'b'):
        raise argparse.ArgumentTypeError("Version must be a or b")
    return x

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=check_type, required=True, help='model version')
args = parser.parse_args()

print('--- argument accepted! version "{}" chosen'.format(args.version))

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)


# ------------------ DATA LOADING ------------------
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

# Set train, eval and test split
n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)


# ------------------ CLASSES AND FUNCTIONS NECESSARY FOR THE APPLICATION ------------------
# class to deal with windows in the temperature and humidity forecasting
class WindowGenerator:
    def __init__(self, input_width, output_steps, mean, std):
        self.input_width = input_width
        self.output_steps = output_steps
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :self.input_width, :]
        labels = features[:, -self.output_steps:, :]

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.output_steps, 2])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width+self.output_steps,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

# Class to deal with two values of MAE (temperature and humidity)
class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=(2,))
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=[0,1])
        self.total.assign_add(error)
        self.count.assign_add(1.)

        return

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)

        return result

# Class to define customed callback to monitor t_mae and h_mae during training at each epoch
class CustomCallback(tf.keras.callbacks.Callback): 
    
    def on_epoch_end(self, epoch, logs=None):
        t = logs["mean_absolute_error"][0]
        h = logs["mean_absolute_error"][1]
        print("\t T mae={:.3f}, H mae={:.3f}".format(t, h))
    

# Function to load and evaluate the final tflite.zlib model
def load_and_evaluation(path, dataset):
    f = open(path, 'rb')
    decompressed_model = zlib.decompress(f.read())
    interpreter = tf.lite.Interpreter(model_content=decompressed_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # set batch size to 1 when running inference with TFLite models
    dataset = dataset.unbatch().batch(1)
    
    outputs = []
    labels = []
    
    for data in dataset:
        my_input = np.array(data[0], dtype = np.float32)
        label = np.array(data[1], dtype = np.float32)
        labels.append(label)

        interpreter.set_tensor(input_details[0]['index'], my_input)
        interpreter.invoke()
        my_output = interpreter.get_tensor(output_details[0]['index'])
        
        outputs.append(my_output[0])

    outputs = np.array(outputs)
    labels = np.squeeze(np.array(labels))
    
    mae = np.sum(np.sum(np.absolute(outputs - labels), axis = 0), axis = 0)/(labels.shape[0]*output_steps)
    return mae

# Function for weight + activations post-trianing quantization  
def representative_dataset_generator():
    for x, _ in train_ds.take(1000):
        yield [x]

# ------------------ CREATE DATASETS ------------------
print('--- 1) dataset creation...')

input_width = 6

# Parameters for dataset generation for each version
if args.version == "a":
    output_steps = 3
else:
    output_steps = 9

# Create train, validation and test dataset (through WindowGenerator class)
generator = WindowGenerator(input_width, output_steps, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)
    

# ------------------ MODELS DEFINTION ------------------
print('--- 2) model definition...')

# Width multiplier for structured pruning for each version 
if args.version == 'a':
    alpha = 0.03
else:
    alpha = 0.04

# Define models including structured pruning via width scaling (width multiplier = alpha) 
pruned_mlp = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(input_width, 2), name='flatten'),
    tf.keras.layers.Dense(int(128*alpha), activation='relu', name='dense1'),
    tf.keras.layers.Dense(int(128*alpha), activation='relu', name='dense2'),
    tf.keras.layers.Dense(units = int(2*output_steps), name='output_layer'),
    tf.keras.layers.Reshape([output_steps, 2])
])

pruned_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv1D(input_shape = (input_width, 2), filters=int(64*alpha), kernel_size=3, activation='relu'),    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=int(2*output_steps)),
    tf.keras.layers.Reshape([output_steps, 2])
])

# Select model to train (from args.version input)
MODELS = {'a': pruned_mlp, 'b': pruned_cnn}
model = MODELS[args.version]

# Define loss, optimizer and metrics for the training procedure
loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()
metrics = [MultiOutputMAE()]


# ------------------ TRAINING WITH MAGNITUED-BASED PRUNING ------------------
epochs = 50

# Define the sparsity scheduler for each version
if args.version == 'a':
    pruning_params = {'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.2,
        final_sparsity=0.4,
        begin_step=2*len(train_ds),
        end_step=20*len(train_ds)
        )
    }
else:
    pruning_params = {'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0,
        final_sparsity=0.25,
        begin_step=2*len(train_ds),
        end_step=20*len(train_ds)
        )
    }

# Make the whole model to train with pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model = prune_low_magnitude(model, **pruning_params)

# Define callbacks
pr_callback = tfmot.sparsity.keras.UpdatePruningStep() # pruning cb
es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=6) # early stopping cb
customed_callback = CustomCallback() # customed cb (for t_mae, h_mae in each epoch)

callbacks = [pr_callback, es_callback, customed_callback]

# Train the model
model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
print('--- 3) training model ...')
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=[callbacks], verbose=2)
print(model.summary())

# Strip the model after training
model = tfmot.sparsity.keras.strip_pruning(model)


# ------------------ POST-TRAINING QUANTIZATION AND CONVERSION TO TFLITE ------------------
print('--- 4) Quantizing trained model...')

# Converting the tf.keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # standard (8-bit) weights-only 

if args.version == "b":
    # weights+activations 8-bits integer quantization
    converter.representative_dataset = representative_dataset_generator

tflite_model = converter.convert()

# ------------------ SAVING AND EVALUATION TFLITE MODEL AS ZLIB (COMPRESSED) ------------------
print('--- 5) saving and evaluating TFLite model...')

if not os.path.exists('./models/'):
    os.makedirs('./models/')

model_dir = os.path.join('.', 'models', 'Group2_th_{}.tflite.zlib'.format(args.version))
with open(model_dir, 'wb') as fp:
    # zlib compression
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)

# Size of the final tflite.zlib model
print('Model size version {}: {:.2f}kB'.format(args.version, os.path.getsize(model_dir)/1000))

# Evaluation of the tflite.zlib model
mae = load_and_evaluation(model_dir, test_ds)
print('MAE of model version {}: [t_mae={:.3f}, h_mae={:.3f}]'.format(args.version, mae[0], mae[1]))