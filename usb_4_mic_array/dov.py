import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import wave
import pickle
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from IPython import display
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier


class Audio:
    '''
    A simple wrapper class for (1-channel) audio data
    data is a 1-D NumPy array containing the data
    rate is a number expressing the samples per second
    '''
    
    def __init__(self, data, rate):
        self.data = data
        self.rate = rate
        
    def play(self):
        return display.Audio(self.data, rate=self.rate)
    
    def plot_wave(self):
        librosa.display.waveplot(self.data, sr=self.rate)
        
    def plot_spectrum(self):
        n_fft = int(self.rate / 20)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.data, n_fft)), ref=np.max)
        librosa.display.specshow(D, y_axis='linear', sr=self.rate, hop_length=n_fft/4)
        
    @classmethod
    def fromfile(cls, fn):
        return cls(*librosa.load(fn, sr=None))
        
        
def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=16):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT) method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(interp * fs)
    
    return tau, cc
    
    
def date_fft(wave_data,framerate):
    # 采样点数，修改采样点数和起始位置进行不同位置和长度的音频波形分析
    N =len(wave_data)
    start = 0  # 开始采样位置
    df = framerate / (N - 1)  # 分辨率
    freq = [df * n for n in range(0, N)]  # N个元素
    print(len(freq))
    wave_data2 = wave_data[start:start + N]
    c = np.fft.rfft(wave_data2) * 2 / N
    # 常规显示采样频率一半的频谱
    d = int(len(c) / 2)
    '''
    # 仅显示频率在4000以下的频谱
    while freq[d] > 4000:
        d -= 10
    '''
    
    sum_back = 0
    sum_font = 0
    for i in range(len(c)):
        if freq[i] > 4000:
            sum_back = sum_back + abs(c[i])**2
        else:
            sum_font = sum_font + abs(c[i])**2
    sum_rate = np.log(sum_back / sum_font)    

    return sum_rate



def get_features(channel_recordings):
    df = pd.DataFrame(columns=col_names)
    data_row = {}
    audio_files = [Audio.fromfile(r) for r in channel_recordings]
    four_channels = [a.data for a in audio_files]
    for i in range(4):
        for j in range(i+1,4):
            gcc_phat_data = gcc_phat(four_channels[i], four_channels[j], 
                                  fs = audio_files[0].rate, max_tau=0.236 * 1e-3, interp=1)
            data_row[f'gccphat_{i}_{j}_peakval'] = gcc_phat_data[1][11]
            data_row[f'gccphat_{i}_{j}_auc'] = np.sum(gcc_phat_data[1])
            data_row[f'gccphat_{i}_{j}_maxshift'] = gcc_phat_data[0]
            for k in range(23):
                data_row[f'gccphatval_{i}_{j}_{k}'] = gcc_phat_data[1][k]
    
    df = df.append(data_row, ignore_index=True)
    return df



channel_recordings1 = []
for i in range(0,5):
    channel_recordings1.append(f'./data/mar06/None_None_None_output_dev0_{i}.wav')
print(channel_recordings1)


channel_recordings2 = []
for i in range(0,5):
    channel_recordings2.append(f'./data/mar06/None_None_None_output_dev1_{i}.wav')
print(channel_recordings2)

audio_files1 = [Audio.fromfile(r) for r in channel_recordings1]
four_channels1 = [a.data for a in audio_files1]

audio_files2 = [Audio.fromfile(r) for r in channel_recordings2]
four_channels2 = [a.data for a in audio_files2]

gcc_phat_data = gcc_phat(four_channels1[0], four_channels2[0], fs = audio_files1[0].rate, max_tau=0.236 * 1e-3, interp=1)
print(gcc_phat_data[0])
    
'''    
channel_recordings1 = []
for i in range(1,5):
    channel_recordings1.append(f'./data/mar06/None_None_None_output_dev0_{i}.wav')
print(channel_recordings1)


channel_recordings2 = []
for i in range(1,5):
    channel_recordings2.append(f'./data/mar06/None_None_None_output_dev1_{i}.wav')
print(channel_recordings2)

col_names = [*[f'gccphat_{i}_{j}_{d}' for i in range(4) for j in range(i+1, 4) for d in ['maxshift', 'auc', 'peakval']],
             *[f'gccphatval_{i}_{j}_{k}' for i in range(4) for j in range(i+1, 4) for k in range(23)]]
print(col_names)



df1 = get_features(channel_recordings1)
x1 = df1.values

df2 = get_features(channel_recordings2)
x2 = df2.values

with open('models/model-90to90.sav', 'rb') as f:
    tc_fitted = pickle.load(f)
y1 = tc_fitted.predict(x1)
y2 = tc_fitted.predict(x2)
print(y1, y2)


audio_files = [Audio.fromfile(r) for r in channel_recordings1]
hlbr_list1 = []
sum1 = 0
for audio_file in audio_files:
    hlbr_list1.append(date_fft(audio_file.data,audio_file.rate))
    sum1 += date_fft(audio_file.data,audio_file.rate)
    
audio_files = [Audio.fromfile(r) for r in channel_recordings2]
hlbr_list2 = []
sum2 = 0
for audio_file in audio_files:
    hlbr_list2.append(date_fft(audio_file.data,audio_file.rate))
    sum2 += date_fft(audio_file.data,audio_file.rate)


print(hlbr_list1)
print(hlbr_list2)

print(sum1 /4, sum2/4)

if sum1/4 > sum2/4:
    print('面向设备0')
else:
    print('面向设备1')
'''
