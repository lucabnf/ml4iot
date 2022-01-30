import argparse 
from scipy import signal
import numpy as np
import os
import zlib
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def check_type(x):
    if (x != 'a') and (x != 'b') and (x != 'c'):
        raise argparse.ArgumentTypeError("Version must be a, b or c ")
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
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

# Set train, eval and test split
kws_train_split = open("kws_train_split.txt", "r")
train_lines = kws_train_split.read().splitlines()
train_files = tf.convert_to_tensor(train_lines)
num_samples_train = train_files.shape

kws_val_split = open("kws_val_split.txt", "r")
val_lines = kws_val_split.read().splitlines()
val_files = tf.convert_to_tensor(val_lines)
num_samples_val = val_files.shape

kws_test_split = open("kws_test_split.txt", "r")
test_lines = kws_test_split.read().splitlines()
test_files = tf.convert_to_tensor(test_lines)
num_samples_test = test_files.shape

num_samples = num_samples_train[0] + num_samples_val[0] + num_samples_test[0]

# Label mapping
labels = open("labels.txt", "r")
labels = str(labels.read())
characters_to_remove = "[]''""  "
for character in characters_to_remove: 
    labels = labels.replace(character, "")
LABELS = labels.split(",")


# ------------------ CLASSES AND FUNCTIONS NECESSARY FOR THE APPLICATION ------------------
# Resampling function
def res(audio, sampling_rate):        
    audio = signal.resample_poly(audio, 1, 16000 // sampling_rate)
    return np.array(audio, dtype = np.float32)

# Translation of the resampling function from a numpy function to a tensorflow function
def tf_function(audio, sampling_rate):
    audio = tf.numpy_function(res, [audio, sampling_rate], tf.float32)
    return audio

# class to read the file audio, to execute the preprocessing and to create the datasets 
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        # resampling 
        if(self.sampling_rate != 16000):
            audio = tf_function(audio, self.sampling_rate)

        audio = tf.squeeze(audio, axis=1)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

# Function to load and evaluate quantized models (.tflite.zlib)
def load_and_evaluation(path, dataset):
    # unzip zlib model
    f = open(path, 'rb')
    decompressed_model = zlib.decompress(f.read())
    # evaluate tflite model
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
    
    acc = sum(np.equal(labels, np.argmax(outputs, axis=1)))/len(outputs)
                 
    return acc

# Function for weight and activations quantization 
def representative_dataset_generator():
    for x, _ in train_ds.take(1000):
        yield [x]
    

# ------------------ PREPROCESSING PARAMETERS FOR EACH VERSION ------------------
if args.version == 'a': # default values
    MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
            'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
            'num_coefficients': 10}
    sampling_rate = 16000 # no resampling

else : # preprocessing params for version b and c
    MFCC_OPTIONS = {'frame_length': 320, 'frame_step': 160, 'mfcc': True,
            'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 16,
            'num_coefficients': 10}
    sampling_rate = 8000
    
options = MFCC_OPTIONS
strides = [2, 1]


# ------------------ CREATE DATASETS ------------------
print('--- 1) dataset creation...')
# Create train, validation and test dataset (through SignalGenerator class)
generator = SignalGenerator(LABELS, sampling_rate, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)


# ------------------ MODEL DEFINTION ------------------
print('--- 2) model definition...')

# Width multiplier for structured pruning for each version 
if args.version == 'a':
    alpha = 1
elif args.version == 'b':
    alpha = 0.7
else:
    alpha = 0.3
    
# DS-CNN (depthwise separable convolutional neural network)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=[3,3], strides=strides, use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
    tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=[1,1], strides=[1,1], use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
    tf.keras.layers.Conv2D(filters=int(256*alpha), kernel_size=[1,1], strides=[1,1], use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units = 8)
])

# Define loss, optimizer and metrics for the training procedure
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

# ------------------ TRAINING WITH MAGNITUED-BASED PRUNING ------------------
# Define the sparsity scheduler for each version
if args.version == 'a':
    epochs = 25
    pruning_params = {'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.25,
        final_sparsity=0.75,
        begin_step=2*len(train_ds),
        end_step=20*len(train_ds)
        )
    }
elif args.version == 'b':
    epochs = 20
    pruning_params = {'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.40,
        final_sparsity=0.75,
        begin_step=2*len(train_ds),
        end_step=22*len(train_ds)
        )
    }
else:
    epochs = 25
    pruning_params = {'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.15,
        final_sparsity=0.35,
        begin_step=2*len(train_ds),
        end_step=30*len(train_ds)
        )
    }

# Make the whole model to train with pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model = prune_low_magnitude(model, **pruning_params)

# Define the pruning callback
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

# Train the model
input_shape = [1, 49, 10, 1]
model.build(input_shape) 
model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
print('--- 3) training model for', epochs ,'epochs...')
model.fit(train_ds, epochs=epochs,validation_data=val_ds, callbacks=callbacks, verbose=2)
print(model.summary())

# Strip the model after training
model = tfmot.sparsity.keras.strip_pruning(model)


# ------------------ POST-TRAINING QUANTIZATION AND CONVERSION TO TFLITE ------------------
print('--- 4) Quantizing trained model...')

# Converting the tf.keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] # standard (8-bit) weights-only

if args.version == 'a':
    # weights-only 16-bits float
    converter.target_spec.supported_types = [tf.float16]
elif args.version == 'b':
    # weights+activations 8-bits integer quantization
    converter.representative_dataset = representative_dataset_generator
else:
    # weights-only 16-bits float
    converter.target_spec.supported_types = [tf.float16]
    
tflite_model = converter.convert()

# ------------------ SAVING AND EVALUATION TFLITE MODEL AS ZLIB (COMPRESSED) ------------------
print('--- 5) saving and evaluating TFLite model...')

if not os.path.exists('./models/'):
    os.makedirs('./models/')

model_dir = os.path.join('.', 'models', 'Group2_kws_{}.tflite.zlib'.format(args.version))
with open(model_dir, 'wb') as fp:    
    # zlib compression
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)

# Size of the final tflite.zlib model
print('Model size version {}: {:.2f}kB'.format(args.version, os.path.getsize(model_dir)/1000))

# Evaluation of the tflite.zlib model
acc = load_and_evaluation(model_dir, test_ds)
print('Accuracy of model version {} = {:.3f}'.format(args.version, acc))