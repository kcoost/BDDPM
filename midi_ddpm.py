# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import os
from pretty_midi import PrettyMIDI
import pypianoroll
import matplotlib.pyplot as plt
import numpy as np
import torch
import glob

class MidiClass:
    def __init__(self):
        pass
    
    def read(self, file_name):
        self.multitrack = pypianoroll.read(file_name)
    
    def trim(self, start, end):
        assert start < end
        self.multitrack.trim(start * self.multitrack.resolution, end * self.multitrack.resolution)
    
    def binarize(self):
        self.multitrack.binarize()
    
    def to_wav(self):
        prettytrack = pypianoroll.to_pretty_midi(self.multitrack)
        waveform = prettytrack.synthesize()
        return waveform

    def reduce(self, remove_silent = True):
        self.binarize()
        all_tracks = np.zeros(self.multitrack.tracks[0].pianoroll.shape)
        for track in self.multitrack.tracks:
            if not track.is_drum:
                all_tracks += track.pianoroll
        
        # remove
        start, end = self.remove_silent(all_tracks)
        all_tracks = np.concatenate((np.zeros((6, 128), dtype=bool), all_tracks[start:end],
                                     np.zeros((6, 128), dtype=bool)), axis = 0)
        self.multitrack.tracks = [pypianoroll.BinaryTrack(name = 'Combined', program=0, is_drum=False, pianoroll=all_tracks)]
    
    def remove_silent(self, all_tracks):
        x = all_tracks.sum(1)
        start = sum(np.cumsum(x) == 0)
        end = len(x) - sum(np.cumsum(x[::-1]) == 0)
        return start, end


def midi_to_pianoroll(file_name):
    midi = MidiClass()
    midi.read(file_name)
    midi.reduce()
    return midi.multitrack.tracks[0].pianoroll

def main(args):
    file_names = glob.glob(os.path.join(args.midi_path, '*.mid'))
    try:
        os.mkdir(args.tensor_path)
    except:
        pass

    for file_name in file_names:
        try:
            pianoroll = midi_to_pianoroll(file_name)
            pianoroll = torch.BoolTensor(pianoroll)
            torch.save(pianoroll, os.path.join(args.tensor_path, os.path.split(file_name)[-1].replace('.mid', '.pt')))
        except:
            print(f"Couldn't convert {os.path.split(file_name)[-1]}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--midi_path', type = str, default = 'asd', help='N')
    parser.add_argument('--tensor_path', type = str, default = 'qwe', help='N')
    args = parser.parse_args()
    
    main(args)