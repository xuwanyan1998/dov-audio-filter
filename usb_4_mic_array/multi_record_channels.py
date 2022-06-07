import pyaudio
import wave
import numpy as np
import argparse



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

for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
    data0 = stream0.read(CHUNK)
    data1 = stream1.read(CHUNK)
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


print("* done recording")

stream0.stop_stream()
stream1.stop_stream()
stream0.close()
stream1.close()
p.terminate()

frames0 = [frames0_0, frames0_1, frames0_2, frames0_3, frames0_4, frames0_5]
frames1 = [frames1_0, frames1_1, frames1_2, frames1_3, frames1_4, frames1_5]

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

