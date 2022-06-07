import librosa
import soundfile
import os


filedir = './data/mar06'
filename_list = os.listdir(filedir)
for filename in filename_list:
    file_path = os.path.join(filedir,filename) 
    y, sr = librosa.load(file_path, sr=16000)  # 读取8k的音频文件
    y_16 = librosa.resample(y, orig_sr=sr, target_sr=48000)  # 采样率转化    
    soundfile.write(file_path, y_16, 48000)	
'''
filename = './data/mydata/None_None_None_output_dev1_4.wav'  # 源文件
newFilename = './data/mydata/None_None_None_output_dev1_4.wav'  # 新采样率保存的文件
 
y, sr = librosa.load(filename, sr=16000)  # 读取8k的音频文件
y_16 = librosa.resample(y, orig_sr=sr, target_sr=48000)  # 采样率转化
 
# 在0.8.0以后的版本，librosa都会将这个函数删除
# librosa.output.write_wav(newFilename, y_16, 16000)
# 推荐用下面的函数进行文件保存
soundfile.write(newFilename, y_16, 48000)  # 重新采样的音频文件保存
'''
