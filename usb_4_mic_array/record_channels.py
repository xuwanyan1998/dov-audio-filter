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
RESPEAKER_INDEX = 0  # refer to input device id
CHUNK = 1024
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = f"./data/{args.people}_{args.room}_{args.angle}_output_"


p = pyaudio.PyAudio()

stream = p.open(
            rate=RESPEAKER_RATE,
            format=p.get_format_from_width(RESPEAKER_WIDTH),
            channels=RESPEAKER_CHANNELS,
            input=True,
            input_device_index=RESPEAKER_INDEX,)

print("* recording")

frames0 = []
frames1 = []
frames2 = []
frames3 = []
frames4 = []
frames5 = []

for i in range(0, int(RESPEAKER_RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    a = np.frombuffer(data, dtype=np.int16)[0::6]
    b = np.frombuffer(data, dtype=np.int16)[1::6]
    c = np.frombuffer(data, dtype=np.int16)[2::6]
    d = np.frombuffer(data, dtype=np.int16)[3::6]
    e = np.frombuffer(data, dtype=np.int16)[4::6]
    f = np.frombuffer(data, dtype=np.int16)[5::6]
    frames0.append(a.tobytes())
    frames1.append(b.tobytes())
    frames2.append(c.tobytes())
    frames3.append(d.tobytes())
    frames4.append(e.tobytes())
    frames5.append(f.tobytes())

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

# channel0
wf = wave.open(WAVE_OUTPUT_FILENAME + '0.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames0))
wf.close()

# channel1
wf = wave.open(WAVE_OUTPUT_FILENAME + '1.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames1))
wf.close()

# channel2
wf = wave.open(WAVE_OUTPUT_FILENAME + '2.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames2))
wf.close()

# channel3
wf = wave.open(WAVE_OUTPUT_FILENAME + '3.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames3))
wf.close()

# channel4
wf = wave.open(WAVE_OUTPUT_FILENAME + '4.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames4))
wf.close()

# channel5
wf = wave.open(WAVE_OUTPUT_FILENAME + '5.wav', 'wb')
wf.setnchannels(1)
wf.setsampwidth(p.get_sample_size(p.get_format_from_width(RESPEAKER_WIDTH)))
wf.setframerate(RESPEAKER_RATE)
wf.writeframes(b''.join(frames5))
wf.close()
