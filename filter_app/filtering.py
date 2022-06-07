import librosa
import audio_conversion
import soundfile
import sys

fnames0 = ["recording0_225_%d.wav" % i for i in range(1,5) ]

fnames = ["s9_upstairs_nowall_trial1_A0_1_0_recording0_0_1-s2_upstairs_nowall_trial2_B0_3_0_recording0_180_1.wav",
  "s9_upstairs_nowall_trial1_A0_1_0_recording0_0_2-s2_upstairs_nowall_trial2_B0_3_0_recording0_180_2.wav",
  "s9_upstairs_nowall_trial1_A0_1_0_recording0_0_3-s2_upstairs_nowall_trial2_B0_3_0_recording0_180_3.wav",
  "s9_upstairs_nowall_trial1_A0_1_0_recording0_0_4-s2_upstairs_nowall_trial2_B0_3_0_recording0_180_4.wav"]

fnames2 = ["s2_upstairs_nowall_trial1_B0_3_0_recording1_225_1-s9_upstairs_nowall_trial1_A0_1_0_recording0_0_1.wav",
  "s2_upstairs_nowall_trial1_B0_3_0_recording1_225_2-s9_upstairs_nowall_trial1_A0_1_0_recording0_0_2.wav",
  "s2_upstairs_nowall_trial1_B0_3_0_recording1_225_3-s9_upstairs_nowall_trial1_A0_1_0_recording0_0_3.wav",
  "s2_upstairs_nowall_trial1_B0_3_0_recording1_225_4-s9_upstairs_nowall_trial1_A0_1_0_recording0_0_4.wav"]

fnames_whats_up_gamers = ["whats_up_gamers_v3/whats-up-gamers-0%d.wav" % i for i in range(1,5) ]

fnames_podcast = ["podcast_gone_wrong/podcast_gone_wrong-0%d.wav" % i for i in range(1,5) ]

# print(fnames_whats_up_gamers)
audio_loads = [ librosa.load(fn, sr=None) for fn in fnames_whats_up_gamers ]

fs = audio_loads[0][1]
audio_data = [ entry[0] for entry in audio_loads ]
max_tau = 0.236e-3

try:
    frame_timelen = int(sys.argv[1]) * 1e-3
except: 
    frame_timelen = 400e-3

new_signals = audio_conversion.process_signal(audio_data,
                                              frame_timelen, fs, max_tau)

try:
    prefix = sys.argv[2]
except:
    prefix = 'conversion_output' 

for i in range(len(new_signals)):
    soundfile.write(f'{prefix}_{i}.wav', new_signals[i], fs)
