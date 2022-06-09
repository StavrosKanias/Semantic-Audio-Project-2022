from thumbnailing import audio_thumb
import matplotlib.pyplot as plt
import libfmp.b
import libfmp.c2
import libfmp.c3
import libfmp.c4
import libfmp.c6
import librosa.display
import librosa
from matplotlib import pyplot as plt
import numpy as np
import msaf

# Uncomment only if you are running Linux
# from essentia.standard import MFCC, FrameGenerator, Windowing, PowerSpectrum, MelBands, UnaryOperator, DCT, MonoLoader
# from essentia import array


def compute_sm_dot(X, Y):
    S = np.dot(np.transpose(X), Y)
    return S


def plot_feature_ssm(X, Fs_X, S, Fs_S,
                     title='', label='Time (seconds)', time=True):
    cmap = libfmp.b.compressed_gray_cmap(alpha=-50)
    libfmp.b.plot_matrix(S, Fs=Fs_S, cmap=cmap, title='',
                         xlabel='', ylabel='', colorbar=True, figsize=(15, 10))


def feature_ssm(F, Fs, title=''):
    N, H = 4096, 1024
    X, Fs_X = libfmp.c3.smooth_downsample_feature_sequence(
        F, Fs/H, filt_len=41, down_sampling=10)
    X = libfmp.c3.normalize_feature_sequence(X, norm='2', threshold=0.001)
    S = compute_sm_dot(X, X)
    plot_feature_ssm(X, 1, S, 1, title=title, label='Time (frames)')
    plt.show()
    plt.savefig(title + '.png')


def plotChromagram(filename, Fs):

    # Load wav
    x, Fs = librosa.load(filename, sr=Fs)
    eps = np.finfo(float).eps
    N, H = 4096, 1024

    C = librosa.feature.chroma_stft(
        y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    plt.figure(figsize=(8, 2))
    librosa.display.specshow(10 * np.log10(eps + C), x_axis='time',
                             y_axis='chroma', sr=Fs, hop_length=H)
    plt.colorbar()
    plt.savefig('Chromagram.png')
    plt.xlabel('Time(minutes)')
    plt.ylabel('Equal Temperament notes')
    plt.show()

    return C, Fs


def plotTempogram(filename, Fs):
    # Compute local onset autocorrelation
    y, sr = librosa.load(filename, sr=Fs)
    hop_length = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    T = librosa.feature.tempogram(onset_envelope=oenv, sr=sr,
                                  hop_length=hop_length)
    # Compute global onset autocorrelation
    ac_global = librosa.autocorrelate(oenv, max_size=T.shape[0])
    ac_global = librosa.util.normalize(ac_global)
    # Estimate the global tempo for display purposes
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr,
                               hop_length=hop_length)[0]
    fig, ax = plt.subplots(nrows=3, figsize=(10, 10))
    times = librosa.times_like(oenv, sr=sr, hop_length=hop_length)
    ax[0].plot(times, oenv, label='Onset strength')
    ax[0].label_outer()
    ax[0].legend(frameon=True)
    librosa.display.specshow(T, sr=sr, hop_length=hop_length,
                             x_axis='time', y_axis='tempo', cmap='magma',
                             ax=ax[1])
    ax[1].axhline(tempo, color='w', linestyle='--', alpha=1,
                  label='Estimated tempo={:g}'.format(tempo))
    ax[1].legend(loc='upper right')
    ax[1].set(title='Tempogram')
    ax[1].set_xlabel('Time(minutes)')

    x = np.linspace(0, T.shape[0] * float(hop_length) / sr,
                    num=T.shape[0])

    freqs = librosa.tempo_frequencies(
        T.shape[0], hop_length=hop_length, sr=sr)
    ax[2].semilogx(freqs[1:], np.mean(T[1:], axis=1),
                   label='Mean local autocorrelation', base=2)
    ax[2].semilogx(freqs[1:], ac_global[1:], '--', alpha=0.75,
                   label='Global autocorrelation', base=2)
    ax[2].axvline(tempo, color='black', linestyle='--', alpha=.8,
                  label='Estimated tempo={:g}'.format(tempo))
    ax[2].legend(frameon=True)
    ax[2].set(xlabel='BPM')
    ax[2].grid(True)
    plt.savefig('Tempogram.png')
    plt.show()

    return T, Fs


# def plotCepstrum(filename, Fs):

#     loader = MonoLoader(filename=filename, sampleRate=Fs)
#     audio = loader()
#     w = Windowing(type='hann')
#     spectrum = PowerSpectrum()
#     spec = spectrum(w(audio))
#     inputSize = spec.size
#     f_low = 50
#     f_high = Fs / 2 - 1  # Nyquist frequency
#     n_bands = 250
#     melbands = MelBands(lowFrequencyBound=f_low,
#                         highFrequencyBound=f_high,
#                         inputSize=inputSize,
#                         numberBands=n_bands,
#                         type='magnitude',  # We already computed the power.
#                         sampleRate=Fs)
#     mels = melbands(spec)
#     log10 = UnaryOperator(type='log10')
#     log_mels = log10(mels)

#     # Perform a DCT
#     dct = DCT(inputSize=n_bands, outputSize=n_bands)
#     mfccs = dct(log_mels)

#     plt.figure()
#     plt.bar(np.arange(len(mfccs)), mfccs, align='center')
#     plt.title('Mel-Frequency Cepstrum')
#     plt.xlabel('Cosines')
#     plt.xlim(-1, 10)
#     plt.ylabel('Coefficients')
#     plt.savefig('Cepstrum.png')


# def plotMFCCs(filename, Fs):
#     N, H = 4096, 1024
#     loader = MonoLoader(filename=filename, sampleRate=Fs)
#     audio = loader()
#     MFCCS = librosa.feature.mfcc(y=audio, sr=Fs, hop_length=H, n_fft=N)
#     # Plot MFCCs
#     plt.figure()
#     plt.imshow(MFCCS,
#                aspect='auto',
#                origin='lower',
#                interpolation='none',
#                cmap=plt.cm.Blues)
#     plt.title("MFCCs")
#     plt.xlabel("Frames")
#     plt.ylabel("MFCCs")
#     plt.savefig('MFCCs.png')
#     return MFCCS, Fs


def extract_thumbnail(filename, duration):
    thumbnailer = audio_thumb(filename)
    thumbnailer.thumb_time(duration)


def main():
    filename = 'Huseyni_Saz_Semaisi _Yiannis_Balafoutis_(Ney).wav'
    Fs = 44100
    # plotCepstrum(filename, Fs)
    # mfccs, Fs = plotMFCCs(filename, Fs)
    # feature_ssm(mfccs, Fs, 'SSM form MFCCs')
    C, Fs = plotChromagram(filename, Fs)
    feature_ssm(C, Fs, 'SSM from Chroma')
    T, Fs = plotTempogram(filename, Fs)
    feature_ssm(T, Fs, 'SSM from Tempo')
    extract_thumbnail(filename, 20)


if __name__ == "__main__":
    main()
