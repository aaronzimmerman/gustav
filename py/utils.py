from nupic.encoders import AdaptiveScalarEncoder


# first arg is how many bits represent each input,
import numpy
from py.note import Note

pitch_encode_length = 20
duration_encode_length = 30
# n is how large the array is total
pitch_encoder = AdaptiveScalarEncoder(7, n=pitch_encode_length, name='pitch')
duration_encoder = AdaptiveScalarEncoder(7, n=duration_encode_length, name='duration')


def note_to_sdr(note):

    first_part = pitch_encoder.encode(note.pitch)
    second_part = duration_encoder.encode(note.duration)


    return numpy.append(first_part, second_part)



def sdr_to_note(sdr):
    #({'pitch': ([[332.0, 332.0]], '332.00')}, ['pitch'])

    first_part = sdr[:pitch_encode_length]
    second_part = sdr[pitch_encode_length+1:]

    [[pitch]], pitch_str = pitch_encoder.decode(first_part)['pitch']
    [[duration]], dur_str = duration_encoder.decode(second_part)['duration']

    return Note(pitch=pitch[0], duration = duration[0])







