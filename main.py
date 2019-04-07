import librosa

import os
import pandas as pd
import glob 
import IPython.display as ipd
import matplotlib.pyplot as plt
C = 'data/notes/C.wav'
ipd.Audio(C)

data, sampling_rate = librosa.load(C)
plt.figure(figsize=(12, 4))
librosa.display.waveplot(data, sr=sampling_rate)