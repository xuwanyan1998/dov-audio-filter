import pyaudio
import wave
import numpy as np
import argparse
import librosa
import soundfile
import os
import pandas as pd
import librosa.display
import pickle
import requests
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from IPython import display
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier


parser = argparse.ArgumentParser(description="Demo of argparse")
parser.add_argument('-p','--people', help="people")
parser.add_argument('-r','--room', help='room')
parser.add_argument('-a','--angle', help='angle')
args = parser.parse_args()

p = pyaudio.PyAudio()
for i in range(p.get_device_count()):
   dev = p.get_device_info_by_index(i)
   print((i,dev['name'],dev['maxInputChannels']))

RESPEAKER_RATE = 16000
RESPEAKER_CHANNELS = 6 # change base on firmwares, default_firmware.bin as 1 or i6_firmware.bin as 6
RESPEAKER_WIDTH = 2
# run getDeviceInfo.py to get index
RESPEAKER_INDEX_0 = 4  # refer to input device id
RESPEAKER_INDEX_1 = 5
RESPEAKER_INDEX_2 = 6
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = f"./data/mar06/{args.people}_{args.room}_{args.angle}_output_"


p = pyaudio.PyAudio()

stream0 = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX_0,)
            
stream1 = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX_1,)

stream2 = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX_2,)


print("* recording")

frames0_0 = []
frames0_1 = []
frames0_2 = []
frames0_3 = []
frames0_4 = []
frames0_5 = []

frames1_0 = []
frames1_1 = []
frames1_2 = []
frames1_3 = []
frames1_4 = []
frames1_5 = []

frames2_0 = []
frames2_1 = []
frames2_2 = []
frames2_3 = []
frames2_4 = []
frames2_5 = []

for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
    data0 = stream0.read(CHUNK)
    data1 = stream1.read(CHUNK)
    data2 = stream2.read(CHUNK)
    a0 = np.frombuffer(data0, dtype=np.int16)[0::6]
    b0 = np.frombuffer(data0, dtype=np.int16)[1::6]
    c0 = np.frombuffer(data0, dtype=np.int16)[2::6]
    d0 = np.frombuffer(data0, dtype=np.int16)[3::6]
    e0 = np.frombuffer(data0, dtype=np.int16)[4::6]
    f0 = np.frombuffer(data0, dtype=np.int16)[5::6]
    a1 = np.frombuffer(data1, dtype=np.int16)[0::6]
    b1 = np.frombuffer(data1, dtype=np.int16)[1::6]
    c1 = np.frombuffer(data1, dtype=np.int16)[2::6]
    d1 = np.frombuffer(data1, dtype=np.int16)[3::6]
    e1 = np.frombuffer(data1, dtype=np.int16)[4::6]
    f1 = np.frombuffer(data1, dtype=np.int16)[5::6]
    a2 = np.frombuffer(data2, dtype=np.int16)[0::6]
    b2 = np.frombuffer(data2, dtype=np.int16)[1::6]
    c2 = np.frombuffer(data2, dtype=np.int16)[2::6]
    d2 = np.frombuffer(data2, dtype=np.int16)[3::6]
    e2 = np.frombuffer(data2, dtype=np.int16)[4::6]
    f2 = np.frombuffer(data2, dtype=np.int16)[5::6]
    frames0_0.append(a0.tobytes())
    frames0_1.append(b0.tobytes())
    frames0_2.append(c0.tobytes())
    frames0_3.append(d0.tobytes())
    frames0_4.append(e0.tobytes())
    frames0_5.append(f0.tobytes())
    frames1_0.append(a1.tobytes())
    frames1_1.append(b1.tobytes())
    frames1_2.append(c1.tobytes())
    frames1_3.append(d1.tobytes())
    frames1_4.append(e1.tobytes())
    frames1_5.append(f1.tobytes())
    frames2_0.append(a2.tobytes())
    frames2_1.append(b2.tobytes())
    frames2_2.append(c2.tobytes())
    frames2_3.append(d2.tobytes())
    frames2_4.append(e2.tobytes())
    frames2_5.append(f2.tobytes())


print("* done recording")

stream0.stop_stream()
stream1.stop_stream()
stream2.stop_stream()
stream0.close()
stream1.close()
stream2.close()
p.terminate()

frames0 = [frames0_0, frames0_1, frames0_2, frames0_3, frames0_4, frames0_5]
frames1 = [frames1_0, frames1_1, frames1_2, frames1_3, frames1_4, frames1_5]
frames2 = [frames2_0, frames2_1, frames2_2, frames2_3, frames2_4, frames2_5]

for i in range(6):
   wf = wave.open(WAVE_OUTPUT_FILENAME + f'dev0_{i}.wav', 'wb')
   wf.setnchannels(1)
   wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
   wf.setframerate(RESPEAKER_RATE)
   wf.writeframes(b''.join(frames0[i]))
   wf.close() 
   
   
for i in range(6):
   wf = wave.open(WAVE_OUTPUT_FILENAME + f'dev1_{i}.wav', 'wb')
   wf.setnchannels(1)
   wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
   wf.setframerate(RESPEAKER_RATE)
   wf.writeframes(b''.join(frames1[i]))
   wf.close()    

for i in range(6):
   wf = wave.open(WAVE_OUTPUT_FILENAME + f'dev2_{i}.wav', 'wb')
   wf.setnchannels(1)
   wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
   wf.setframerate(RESPEAKER_RATE)
   wf.writeframes(b''.join(frames2[i]))
   wf.close() 

'''
filedir = './data/mar06'
filename_list = os.listdir(filedir)
for filename in filename_list:
    file_path = os.path.join(filedir,filename) 
    y, sr = librosa.load(file_path, sr=16000)  # 读取8k的音频文件
    y_16 = librosa.resample(y, orig_sr=sr, target_sr=48000)  # 采样率转化    
    soundfile.write(file_path, y_16, 48000)
'''
    
    



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
            print(i, j)       
            gcc_phat_data = gcc_phat(four_channels[i], four_channels[j], 
                                  fs = audio_files[0].rate, max_tau=0.236 * 1e-3, interp=1)
            data_row[f'gccphat_{i}_{j}_peakval'] = gcc_phat_data[1][11]
            data_row[f'gccphat_{i}_{j}_auc'] = np.sum(gcc_phat_data[1])
            data_row[f'gccphat_{i}_{j}_maxshift'] = gcc_phat_data[0]
            print(gcc_phat_data[0])
            for k in range(23):
                data_row[f'gccphatval_{i}_{j}_{k}'] = gcc_phat_data[1][k]
    
    df = df.append(data_row, ignore_index=True)
    return df
    
    
channel_recordings1 = []
for i in range(1,5):
    channel_recordings1.append(f'./data/mar06/None_None_None_output_dev0_{i}.wav')


channel_recordings2 = []
for i in range(1,5):
    channel_recordings2.append(f'./data/mar06/None_None_None_output_dev1_{i}.wav')


channel_recordings3 = []
for i in range(1,5):
    channel_recordings3.append(f'./data/mar06/None_None_None_output_dev2_{i}.wav')


col_names = [*[f'gccphat_{i}_{j}_{d}' for i in range(4) for j in range(i+1, 4) for d in ['maxshift', 'auc', 'peakval']],
             *[f'gccphatval_{i}_{j}_{k}' for i in range(4) for j in range(i+1, 4) for k in range(23)]]



'''
df1 = get_features(channel_recordings1)
x1 = df1.values

df2 = get_features(channel_recordings2)
x2 = df2.values

df3 = get_features(channel_recordings3)
x3 = df3.values

with open('models/model-90to90.sav', 'rb') as f:
    tc_fitted = pickle.load(f)
y1 = tc_fitted.predict(x1)
y2 = tc_fitted.predict(x2)
y3 = tc_fitted.predict(x3)
print(y1, y2, y3)
'''

audio_files = [Audio.fromfile(r) for r in channel_recordings1]
hlbr_list1 = []
sum1 = 0
for audio_file in audio_files:
    x = date_fft(audio_file.data,audio_file.rate)
    hlbr_list1.append(x)
    sum1 += x
    
audio_files = [Audio.fromfile(r) for r in channel_recordings2]
hlbr_list2 = []
sum2 = 0
for audio_file in audio_files:
    x = date_fft(audio_file.data,audio_file.rate)
    hlbr_list2.append(x)
    sum2 += x


audio_files = [Audio.fromfile(r) for r in channel_recordings3]
hlbr_list3 = []
sum3 = 0
for audio_file in audio_files:
    x = date_fft(audio_file.data,audio_file.rate)
    hlbr_list3.append(x)
    sum3 += x

print(hlbr_list1)
print(hlbr_list2)
print(hlbr_list3)

print(sum1/4, sum2/4, sum3/4)


if sum1/4 > sum2/4 and sum1/4 > sum3/4:
    print('面向设备0')
elif sum2/4 > sum1/4 and sum2/4 > sum3/4:
    print('面向设备1')
else:
    print('面向设备2')




'''
if sum1/4 > sum2/4 and sum1/4 > sum3/4:
    print('面向设备0')
    datas = {'queryID':'123456', 'device':0}
    try:
    	r = requests.get('http://127.0.0.1:8066/awe-control-demo/api', params=datas)
    	print(r.status_code)
    except:
    	print('未开启IOT')
    	print(r.status_code)
elif sum2/4 > sum1/4 and sum2/4 > sum3/4:
    print('面向设备1')
    datas = {'queryID':'123456', 'device':2}
    try:
    	r = requests.get('http://127.0.0.1:8066/awe-control-demo/api', params=datas)
    	print(r.status_code)
    except:
    	print('未开启IOT')
    	print(r.status_code)
else:
    datas = {'queryID':'123456', 'device':1}
    try:
    	r = requests.get('http://127.0.0.1:8066/awe-control-demo/api', params=datas)
    	print(r.status_code)
    except:
    	print('未开启IOT')
    	r = requests.get('http://127.0.0.1:8066/awe-control-demo/api', params=datas)
    	print(r.status_code)
    print('面向设备2')
'''












   
'''
# channel0
wf = wave.open(WAVE_OUTPUT_FILENAME + '0.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames0_0))
wf.close()

# channel1
wf = wave.open(WAVE_OUTPUT_FILENAME + '1.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames0_1))
wf.close()

# channel2
wf = wave.open(WAVE_OUTPUT_FILENAME + '2.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames0_2))
wf.close()

# channel3
wf = wave.open(WAVE_OUTPUT_FILENAME + '3.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames0_3))
wf.close()

# channel4
wf = wave.open(WAVE_OUTPUT_FILENAME + '4.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames0_4))
wf.close()

# channel5
wf = wave.open(WAVE_OUTPUT_FILENAME + '5.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames0_5))
wf.close()
'''

