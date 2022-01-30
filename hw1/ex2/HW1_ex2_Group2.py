import os
import tensorflow as tf
import time
import numpy as np
from scipy import signal

# guarantee stable measurements of the execution time
from subprocess import Popen
Popen('sudo sh -c "echo performance >" /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
      shell=True).wait()

def read_file(filename):
    audio = tf.io.read_file("./yes_no/" + filename)
    tf_audio, rate = tf.audio.decode_wav(audio)
    tf_audio = tf.squeeze(tf_audio, 1)

    return rate, tf_audio

def resample(tf_audio, rate, up, down):
    audio = signal.resample_poly(tf_audio, up, down)
    tf_audio = tf.convert_to_tensor(audio, dtype=np.float32)
    rate = tf.convert_to_tensor(np.array(rate * up)/down, dtype=np.int32)

    return rate, tf_audio

def compute_stft(tf_audio, frame_length, frame_step):
    stft = tf.signal.stft(tf_audio,frame_length=frame_length, frame_step=frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)
    return spectrogram

def compute_mfcc(spectrogram, linear_to_mel_weight_matrix, num_mfccs):
    mel_spectrogram = tf.tensordot(spectrogram,linear_to_mel_weight_matrix,1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :num_mfccs]
    return mfcc


# --- PARAMETERS OF THE MFCC SLOW ---
# Parameters for the Spectrogram
length = 16*1e-3
step = 8*1e-3

# Parameters for MFCC
n_mel_bins = 40
num_mfccs = 10
lwr_freq = 20
up_freq = 4000

# MFCC SLOW
mfccs_slow = []
time_slow = []
first_iteration = True
for filename in os.listdir("yes_no"):
    start = time.time()

    rate, tf_audio = read_file(filename)

    if first_iteration is True:
        rate = np.array(rate)
        frame_length = (rate * length).astype(np.int32)
        frame_step = (rate * step).astype(np.int32)

    spectrogram = compute_stft(tf_audio, frame_length=frame_length, frame_step=frame_step)

    if first_iteration is True:
        n_spectrogram_bins = spectrogram.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=n_mel_bins,
                                                                        num_spectrogram_bins=n_spectrogram_bins,
                                                                        sample_rate=rate,
                                                                        lower_edge_hertz=lwr_freq,
                                                                        upper_edge_hertz=up_freq)
        first_iteration = False

    mfcc = compute_mfcc(spectrogram, linear_to_mel_weight_matrix=linear_to_mel_weight_matrix, num_mfccs=num_mfccs)

    mfccs_slow.append(mfcc)

    end = time.time()
    time_slow.append(end - start)


# --- PARAMETERS OF THE MFCC FAST ---
# Parameters for the Spectrogram
length = 16*1e-3
step = 8*1e-3

# Parameters for MFCC
n_mel_bins = 32
num_mfccs = 10
lwr_freq = 20
up_freq = 2000

mfccs_fast = []
time_fast = []
first_iteration = True
for filename in os.listdir("yes_no"):
    start = time.time()

    rate, tf_audio = read_file(filename)
    new_rate, tf_audio = resample(tf_audio, rate=rate, up=1, down=4)

    if first_iteration is True:
        new_rate = np.array(new_rate)
        frame_length = (new_rate * length).astype(np.int32)
        frame_step = (new_rate * step).astype(np.int32)

    spectrogram = compute_stft(tf_audio, frame_length=frame_length, frame_step=frame_step)

    if first_iteration is True:
        n_spectrogram_bins = spectrogram.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins=n_mel_bins,
                                                                            num_spectrogram_bins=n_spectrogram_bins,
                                                                            sample_rate=new_rate,
                                                                            lower_edge_hertz=lwr_freq,
                                                                            upper_edge_hertz=up_freq)
        first_iteration = False

    mfcc = compute_mfcc(spectrogram, linear_to_mel_weight_matrix=linear_to_mel_weight_matrix, num_mfccs=num_mfccs)

    mfccs_fast.append(mfcc)

    end = time.time()
    time_fast.append(end - start)


print("\n----- OUTPUT -----")
print("MFCC slow = {:.2f} ms".format(np.mean(time_slow) * 1000))
print("MFCC fast = {:.2f} ms".format(np.mean(time_fast) * 1000))

SNR_vec = []
for slow, fast in zip(mfccs_slow, mfccs_fast):
    SNR = 20 * np.log10(np.linalg.norm(slow) / np.linalg.norm(slow - fast + 1.e-6))
    SNR_vec.append(SNR)

print('SNR = {:.2f} dB\n'.format(np.mean(SNR_vec)))