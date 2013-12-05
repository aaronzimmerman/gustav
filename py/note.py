class Note(object):

    def __init__(self, pitch, duration):
        self.pitch = pitch
        self.duration = duration

    def __str__(self):
        return "Note {pitch: %s, duration: %s" % (self.pitch, self.duration)

