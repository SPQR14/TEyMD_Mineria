import librosa
import os
import numpy as np
import pandas as pd 

generos = ['prog', 'salsa', 'electronica']
hop_length = 512
n0 = 9000
n1 = 9100

columnas = ['archivo' ,'zero_cr' ,'spectral_centroid', 'spectral_bw' ,'spectral_rf', 'croma']
for x in range(1, 21):
    columnas.append(f'mfcc_{x}')
columnas.append('BPM')
columnas.append('auto_c')
columnas.append('genero')

df = pd.DataFrame(columns = columnas)
for g in generos:
    for nombre in os.listdir(f'../{g}'):
        cancion = f'../{g}/{nombre}'
        samples, sr = librosa.load(cancion, sr = None, mono = True, offset = 0.0, duration = None)
        nombre = nombre.replace(' ', '')
        zero_crossings = librosa.zero_crossings(samples[n0:n1], pad=False)
        zero_crossings = sum(zero_crossings)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y = samples, sr = sr))
        spectral_bw = np.mean(librosa.feature.spectral_bandwidth(y = samples, sr = sr))
        spectral_rf = np.mean(librosa.feature.spectral_rolloff(y = samples, sr = sr))
        croma = np.mean(librosa.feature.chroma_stft(y = samples, sr = sr))
        mfcc = librosa.feature.mfcc(y = samples, sr = sr)
        env = librosa.onset.onset_strength(y = samples, sr = sr, hop_length = hop_length)
        tempograma = librosa.feature.tempogram(onset_envelope = env, sr = sr, hop_length = hop_length)
        auto_c = librosa.autocorrelate(env, max_size = tempograma.shape[0])
        auto_c = librosa.util.normalize(auto_c)
        auto_c = np.mean(auto_c)
        BPM = librosa.beat.tempo(onset_envelope = env, sr = sr, hop_length = hop_length)[0]
        x = f'{nombre} {zero_crossings} {spectral_centroid} {spectral_bw} {spectral_rf} {croma}'
        for m in mfcc:
            x += f' {np.mean(m)}'
        x += f' {BPM} {auto_c} {g}'
        d = dict(zip(columnas, x.split()))
        df = df.append(d, ignore_index = True)
        print(u'Añadido el registro de: ' + nombre)

df.to_csv('../data_set/datos_musica.csv', sep = ',', encoding = 'utf8')
