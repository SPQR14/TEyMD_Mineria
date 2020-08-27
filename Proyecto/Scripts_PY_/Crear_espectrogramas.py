import librosa
import librosa.display
import matplotlib.pyplot as plt
import pathlib
import os

generos = ['prog', 'salsa', 'electronica']
plt.figure(figsize=(20, 8))

for g in generos:
    pathlib.Path('..\\epectrogramas\\' + g).mkdir(parents = True, exist_ok = True)
    for nombre in os.listdir('..\\' + g):
        cancion = '..\\' + g + '\\' + nombre
        samples, sr =  librosa.load(cancion, sr = None, mono = True, offset = 0.0, duration = None)
        X = librosa.stft(samples)
        Xdb = librosa.amplitude_to_db(abs(X))
        librosa.display.specshow(Xdb, sr = sr, x_axis = 'time', y_axis ='log')
        plt.title(nombre)
        plt.savefig('..\\espectrogramas\\' + g + '\\' + nombre + '.png')
        plt.clf()