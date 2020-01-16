import librosa
import argparse
import numpy as np
import librosa.display
from augment import SpecAugment
import matplotlib.pyplot as plt


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='./LibriSpeech/', help='path to dataset/dir to look for files')
parser.add_argument('--policy', default='LD', help='augmentation policies - LB, LD, SM, SS')

args = parser.parse_args()


if __name__ == '__main__':
    
    
    # make a list of all training files in the LibriSpeech Dataset
    training_files = librosa.util.find_files(args.dir, ext=['flac'], recurse=True)
    print('Number of Training Files: ', len(training_files))
    
    # Loop over files and apply SpecAugment
    for file in training_files:
        
        # Load the audio file
        audio, sr = librosa.load(file)
        # Extract Mel Spectrogram Features from the audio file
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=256, hop_length=128, fmax=8000)
        librosa.display.specshow(mel_spectrogram, x_axis='time', y_axis='mel', fmax=8000) # Base
        
        # Apply SpecAugment
        apply = SpecAugment(mel_spectrogram, args.policy)
        
        time_warped = apply.time_warp()
        librosa.display.specshow(time_warped[0, :, :, 0].numpy(), x_axis='time', y_axis='mel', fmax=8000) # Time Warped
        
        freq_masked = apply.freq_mask()
        librosa.display.specshow(freq_masked[0, :, :, 0], x_axis='time', y_axis='mel', fmax=8000) # Freq Masked
        
        time_masked = apply.time_mask()
        librosa.display.specshow(time_masked[0, :, :, 0], x_axis='time', y_axis='mel', fmax=8000) # Time Masked
