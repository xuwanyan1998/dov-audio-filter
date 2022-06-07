import os
import numpy as np
import pandas as pd
import librosa
import librosa.display

from IPython import display
from matplotlib import pyplot as plt


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
    def plot_sfft(self):
        n_fft = int(self.rate / 20)
        librosa.stft(y, n_fft=2048, window='hann', center=True, pad_mode='reflect')
    @classmethod
    def fromfile(cls, fn):
        return cls(*librosa.load(fn, sr=None))

    

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
    print(d)
    '''
    # 仅显示频率在4000以下的频谱
    while freq[d] > 4000:
        d -= 10
    '''
    
    sum_back = 0
    for i in range(4000, len(c)):
        sum_back = sum_back + abs(c[i])**2
 
    sum_font = 0
    for i in range(0, 4000):
        sum_font = sum_font + abs(c[i])**2
        
    sum_rate = np.log(sum_back / sum_font)

    return sum_rate
    

def cal_hlbr(dir_list):
    hlbr_list = list()
    for wave_file in dir_list:
        audio_file = Audio.fromfile(wave_file)
        hlbr = date_fft(audio_file.data, audio_file.rate)
        hlbr_list.append(hlbr)
    return hlbr_list
        