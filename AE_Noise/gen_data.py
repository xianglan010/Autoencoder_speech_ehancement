import os
import soundfile
import numpy as np
import time
from scipy import signal
import pickle
import glob
import random


def read_audio(path, target_fs=None):
    (audio, fs) = soundfile.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs

def write_audio(path, audio, sample_rate):
    soundfile.write(file=path, data=audio, samplerate=sample_rate)

def get_amplitude_scaling_factor(s, n, snr, method='rms'):
    original_sn_rms_ratio = rms(s) / rms(n)
    target_sn_rms_ratio =  10. ** (float(snr) / 20.)    # snr = 20 * lg(rms(s) / rms(n))
    signal_scaling_factor = target_sn_rms_ratio / original_sn_rms_ratio
    return signal_scaling_factor

def additive_mixing(s, n):
    mixed_audio = s + n
    alpha = 1. / np.max(np.abs(mixed_audio))
    mixed_audio *= alpha
    s *= alpha
    n *= alpha
    return mixed_audio, s, n, alpha

def rms(y):
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))


noise_list = os.listdir("./noise_60_16k")  
print(noise_list)
fs = 16000

noise_type = list(range(0,60)) * 77
random.shuffle(noise_type)


if __name__ == '__main__':
    
    
    for wav_file in glob.glob(os.path.join("./train_clean",'*.wav')):
        
        (speech_audio, _) = read_audio(wav_file, target_fs = fs)
        
        #random_num = random.randrange(0,len(noise_list))
        num = 0
        noise_use = noise_type[num]
        
        noise_path = os.path.join("./noise_60_16k/"+noise_list[noise_use])
        noise_offset = noise_list[noise_use][:-4][1:]

        print(noise_list[noise_use], os.path.basename(wav_file))
        (noise_audio, _) = read_audio(noise_path, target_fs = fs)

        noise_offset = speech_audio.shape[0]

        if len(noise_audio) < len(speech_audio):
            n_repeat = int(np.ceil(float(len(speech_audio)) / float(len(noise_audio))))
            noise_audio_ex = np.tile(noise_audio, n_repeat)
            noise_audio = noise_audio_ex[0 : len(speech_audio)]

        else:
            noise_audio = noise_audio[0 : noise_offset]

        
        scaler1 = get_amplitude_scaling_factor(speech_audio, noise_audio, snr= 0)
        scaler2 = get_amplitude_scaling_factor(speech_audio, noise_audio, snr= -5)
        scaler3 = get_amplitude_scaling_factor(speech_audio, noise_audio, snr= 5)
        scaler4 = get_amplitude_scaling_factor(speech_audio, noise_audio, snr= 10)
        scaler5 = get_amplitude_scaling_factor(speech_audio, noise_audio, snr= 15)

        speech_1 = speech_audio * scaler1
        speech_2 = speech_audio * scaler2
        speech_3 = speech_audio * scaler3
        speech_4 = speech_audio * scaler4
        speech_5 = speech_audio * scaler5

        noise_audio_1 = noise_audio.copy()
        noise_audio_2 = noise_audio.copy()
        noise_audio_3 = noise_audio.copy()
        noise_audio_4 = noise_audio.copy()
        noise_audio_5 = noise_audio.copy()

        (mixed_audio_1, speech_audio_1, noise_audio_1, alpha_1) = additive_mixing(speech_1, noise_audio_1)
        (mixed_audio_2, speech_audio_2, noise_audio_2, alpha_2) = additive_mixing(speech_2, noise_audio_2)
        (mixed_audio_3, speech_audio_3, noise_audio_3, alpha_3) = additive_mixing(speech_3, noise_audio_3)
        (mixed_audio_4, speech_audio_4, noise_audio_4, alpha_4) = additive_mixing(speech_4, noise_audio_4)
        (mixed_audio_5, speech_audio_5, noise_audio_5, alpha_5) = additive_mixing(speech_5, noise_audio_5)

        write_audio(os.path.join("./train_clean_mix/" + os.path.basename(wav_file) + '_0db' ), speech_audio_1, fs)
        write_audio(os.path.join("./train_clean_mix/" + os.path.basename(wav_file) + '_-5db'), speech_audio_2, fs)
        write_audio(os.path.join("./train_clean_mix/" + os.path.basename(wav_file) + '_5db'), speech_audio_3, fs)
        write_audio(os.path.join("./train_clean_mix/" + os.path.basename(wav_file) + '_10db'), speech_audio_4, fs)
        write_audio(os.path.join("./train_clean_mix/" + os.path.basename(wav_file) + '_15db'), speech_audio_5, fs)
        #write_audio(os.path.join("./train_noise_0db" + "/" + os.path.basename(wav_file)), noise_audio_1, fs)
        #write_audio(os.path.join("./train_noise_-5db" + "/" + os.path.basename(wav_file)), noise_audio_2, fs)
        #write_audio(os.path.join("./train_noise_5db" + "/" + os.path.basename(wav_file)), noise_audio_3, fs)
        #write_audio(os.path.join("./train_noise_10db" + "/" + os.path.basename(wav_file)), noise_audio_4, fs)
        write_audio(os.path.join("./train_mix" + "/" + os.path.basename(wav_file) + '_0db_' + noise_offset), mixed_audio_1, fs)
        write_audio(os.path.join("./train_mix" + "/" + os.path.basename(wav_file) + '_-5db_'+ noise_offset), mixed_audio_2, fs)
        write_audio(os.path.join("./train_mix" + "/" + os.path.basename(wav_file) + '_5db_'+ noise_offset), mixed_audio_3, fs)
        write_audio(os.path.join("./train_mix" + "/" + os.path.basename(wav_file) + '_10db_'+ noise_offset), mixed_audio_4, fs)
        write_audio(os.path.join("./train_mix" + "/" + os.path.basename(wav_file) + '_15db_'+ noise_offset), mixed_audio_5, fs)
        
        num += 1


              

