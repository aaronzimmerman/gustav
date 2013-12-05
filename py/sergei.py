"""
Sergei learns from tuples of (note, duration)
"""


# python

import csv
import datetime
import logging

from nupic.data.datasethelpers import findDataset
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.opfutils import InferenceElement
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager

import midi
import os
from pylab import *


import sys
from note import Note

import sergei_params

SAVE_LOCATION = "/Users/az/nupic-env/gustav/brains/scale_pitch3"
SAVE_LOCATION_2 = "/Users/az/nupic-env/gustav/brains/scale_pitch3_b"

logger = logging.getLogger(__name__)

_METRIC_SPECS = (
    MetricSpec(field='note', metric='multiStep',
               inferenceElement='multiStepBestPredictions',
               params={'errorMetric': 'aae', 'window': 100, 'steps': 1}),)


def createModel():

 # if not os.exis(SAVE_LOCATION):
     model = ModelFactory.create(sergei_params.SERGEI_PARAMS)

    # model =  ModelFactory.loadFromCheckpoint(SAVE_LOCATION)

     model.enableInference({'predictionSteps':[1,2,5,10],
                           'predictedField': 'note'})
     #
     #model.enableInference({'predictionSteps':[1,2,5,10],
     #                      'predictedField': 'duration'})


    #try model.save, model.load

     return model
#


#  855040
def perform(output_file, num_notes):
    model = createModel()

    input = float(60.0)
    output = [input]

    for i in range(num_notes):
        modelInput = {'pitch': input}
        #logger.info("Passing in %s", modelInput["audio"])

        #this is where the magic happens:
        result = model.run(modelInput)

        logger.info("%s -> %s", input, result.inferences['prediction'][0])
        input = float(result.inferences['prediction'][0])
        output.append(input)



def convertMidiToNotes(pattern):
    """
    Pattern is a sequence of midi events
    track on/off note events
    """

    notes = []
    for track in pattern:
        for event in track:

            if isinstance(event, midi.NoteOnEvent):
                #note = Note(event.pitch, event.length)
                note = {'pitch':event.pitch, 'duration':event.length}
                notes.append(note)


    return notes



# python sergei.py practice ../input/scale.mid
def practice(file_name):
    model = createModel()

    #todo: add rhythms

    pattern = midi.read_midifile(file_name)


    metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                      model.getInferenceType())

    logger.info("learning from file %s", file_name)

    predicted = []
    raw = []
    count = 0
    prediction = 0


    notes = convertMidiToNotes(pattern)

    for note in notes:

        #logger.info(note)
        #count += 1
       # pitch = note.pitch

       # logger.info("input: %s, prediction was %s", pitch, prediction)
       # modelInput = {'pitch': pitch, 'duration':note.duration}
        modelInput = {'note': note}
        logger.info("running with input: %s", modelInput)

        result = model.run(modelInput)

        prediction = result.inferences['prediction']
        logger.info(result.inferences)

        result.metrics = metricsManager.update(result)

        raw.append(note)
        predicted.append(prediction)



    #plot the comparison
    #save the two audio files

    # plot actual and predicted
    #indexes = range(100000)
    #plot(indexes, raw, label="actual")
    #hold(True)
    #plot(indexes, predicted, label="predicted")
    #xlabel('time')
    #ylabel('amp')
    #show()

    model.save("%s" % SAVE_LOCATION)


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  if sys.argv[1] == 'practice':
    practice(sys.argv[2])
  elif sys.argv[2] == 'perform':
    perform(sys.argv[2], 30)
