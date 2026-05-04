import mido
from mido import MidiFile

for msg in MidiFile("read_midi/berge.mid").play():
    print(msg)
