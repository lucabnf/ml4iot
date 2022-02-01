from scipy import signal
import numpy as np
import os
import tensorflow as tf


# Resampling function
def res(audio, sampling_rate):        
    audio = signal.resample_poly(audio, 1, 16000 // sampling_rate)
    return np.array(audio, dtype = np.float32)

# Translation of the resampling function from a numpy function to a tensorflow function
def tf_function(audio, sampling_rate):
    audio = tf.numpy_function(res, [audio, sampling_rate], tf.float32)
    return audio


# class used for preprocessing in the FAST client: it read the audio, compute the mfcc 
# and make a dataset  
class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins, lower_frequency, upper_frequency,
            num_coefficients, mfcc):
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


# Compute softmax values for the prediction given as parameter 
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# function that check the if a prediction is confident or not
def success_checker(prediction, threshold):
    softmax_pred = np.squeeze(softmax(prediction))
    top1 = np.sort(softmax_pred)[-1]
    top2 = np.sort(softmax_pred)[-2]
    if (top1-top2) < threshold:
        # non confident prediction 
        return False
    return True
    

# load a tflite model and evaluate its prediction on a dataset give as parameter
def load_and_evaluation(path, dataset):
    # unzip zlib model
    f = open(path, 'rb')
    model = f.read() 
    # evaluate tflite model
    interpreter = tf.lite.Interpreter(model_content = model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # set batch size to 1 when running inference with TFLite models
    dataset = dataset.unbatch().batch(1)
    
    confident_outputs = []
    labels = []
    not_confident_idx = []
    not_confident_labels = []
    
    for i, data in enumerate(dataset):
        my_input = np.array(data[0], dtype = np.float32)
        label = np.array(data[1], dtype = np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], my_input)
        interpreter.invoke()
        my_output = interpreter.get_tensor(output_details[0]['index'])
        
        # checker policy
        if success_checker(my_output, 0.20):
            confident_outputs.append(np.squeeze(my_output))
            labels.append(int(label))
        else:
            not_confident_idx.append(i)
            not_confident_labels.append(int(label))
    
    labels = labels + not_confident_labels
    # print('not confident predictions', len(not_confident_labels))
    if len(not_confident_idx) == 0:
        acc = sum(np.equal(labels, np.argmax(confident_outputs, axis = 1)))/len(confident_outputs)
        print('Accuracy: {:.3f}%'.format(acc*100))
    return confident_outputs, labels, not_confident_idx




# class used for preprocessing in the SLOW server: it read the audio, compute the mfcc 
# and make a dataset  
class SignalGenerator_slow:
    def __init__(self, audio, sampling_rate, frame_length, frame_step,
            num_mel_bins, lower_frequency, upper_frequency,
            num_coefficients):
        self.audio = audio
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = int(frame_length) // 2 + 1
        
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                self.lower_frequency, self.upper_frequency)


    def pad(self, audio_):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio_), dtype=tf.float32)
        audio = tf.concat([audio_, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio_):
        stft = tf.signal.stft(audio_, frame_length=self.frame_length,
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

    def preprocess_with_mfcc(self):
        audio = tf.convert_to_tensor(self.audio)
        audio, _ = tf.audio.decode_wav(audio)
        audio = tf.squeeze(audio, axis=1)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)
        mfccs = tf.reshape(mfccs, [1,49,10,1], name=None)
        return mfccs



# Function to load a tflite model and make a prediction on the preprocessed audio give as parameter  
def predict(path, preprocessed_audio):
    f = open(path, 'rb')
    model = f.read() 
    # evaluate tflite model
    interpreter = tf.lite.Interpreter(model_content = model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], preprocessed_audio)
    interpreter.invoke()
    my_output = interpreter.get_tensor(output_details[0]['index'])
    my_output = np.squeeze(np.array(my_output))
    return list(my_output)