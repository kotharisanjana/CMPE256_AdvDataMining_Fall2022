import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display  

class SoundFeatures:

    def melfrequencySpectogram(self, sig, fs):
        S = librosa.feature.melspectrogram(y=sig, sr=fs)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Frequency Spectogram')
        plt.tight_layout()
        plt.show()

    def shortTimeFourierTransform(self, sig):
        D = np.abs(librosa.stft(sig))**2
        S = librosa.feature.melspectrogram(S=D)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Frequency Spectogram')
        plt.tight_layout()
        plt.show()

    def beatTracker(self, sig, sample_rate, hop_length):
        tempo, beats = librosa.beat.beat_track(y=sig, sr=sample_rate, hop_length=hop_length)
        beat_times = librosa.frames_to_time(beats, sr=sample_rate, hop_length=hop_length)
        cqt = np.abs(librosa.cqt(sig, sr=sample_rate, hop_length=hop_length))
        subseg = librosa.segment.subsegment(cqt, beats, n_segments=2)
        subseg_t = librosa.frames_to_time(subseg, sr=sample_rate, hop_length=hop_length)

        plt.figure(figsize=(10, 4))
        M = librosa.feature.melspectrogram(y=sig, sr=sample_rate, hop_length=hop_length)
        librosa.display.specshow(librosa.power_to_db(M, ref=np.max), y_axis='mel', x_axis='time', hop_length=hop_length)
        librosa.display.specshow(librosa.amplitude_to_db(cqt,ref=np.max),y_axis='cqt_hz', x_axis='time')
        lims = plt.gca().get_ylim()
        plt.vlines(beat_times, lims[0], lims[1], color='lime', alpha=0.9,linewidth=2, label='Beats')
        plt.vlines(subseg_t, lims[0], lims[1], color='linen', linestyle='--',linewidth=1.5, alpha=0.5, label='Sub-beats')
        plt.title('Beat tracker')
        plt.legend(frameon=True, shadow=True)
        plt.tight_layout()
        plt.show()


    def cqt(self, sig, fs, hop_length):
        cqt_ = np.abs(librosa.cqt(sig, sr=fs, hop_length=hop_length))
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(librosa.amplitude_to_db(cqt_,ref=np.max),y_axis='cqt_hz', x_axis='time')
        plt.title('Constant Q Transform')
        plt.tight_layout()
        plt.show()

    
if __name__ == '__main__':
    sf = SoundFeatures()
    for i in range(1, 6):
        file_name = str(i)+'.wav'
        sig, fs = librosa.load(file_name)

        sf.melfrequencySpectogram(sig, fs)
        sf.shortTimeFourierTransform(sig)
        sf.beatTracker(sig, fs, hop_length=512)
        sf.cqt(sig, fs, hop_length=512)





