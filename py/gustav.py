

# python

import csv
import datetime
import logging

from nupic.data.datasethelpers import findDataset
from nupic.frameworks.opf.metrics import MetricSpec
from nupic.frameworks.opf.modelfactory import ModelFactory
from nupic.frameworks.opf.opfutils import InferenceElement
from nupic.frameworks.opf.predictionmetricsmanager import MetricsManager

from pylab import *

import numpy
import scikits
from scikits.audiolab import aiffread
from numpy import array
from scikits.audiolab.pysndfile._sndfile import Sndfile
import sys

import model_params

logger = logging.getLogger(__name__)

# python /Users/az/nupic-env/gustav/py/gustav.py /Users/az/nupic-env/gustav/input/test.aiff
_METRIC_SPECS = (
    MetricSpec(field='amp', metric='multiStep',
               inferenceElement='multiStepBestPredictions',
               params={'errorMetric': 'aae', 'window': 100, 'steps': 1}),
)


def createModel():
  return ModelFactory.create(model_params.MODEL_PARAMS)


def perform(output_file, frames):
    model = createModel()

    model.enableInference({'predictedField': 'amp'})


    input = float(0.1)
    output = [input]

    for i in range(frames):
        modelInput = {'amp': input}
        #logger.info("Passing in %s", modelInput["audio"])

        #this is where the magic happens:
        result = model.run(modelInput)

        logger.info("%s -> %s", input, result.inferences['prediction'])
        input = result.inferences['prediction']
        output.append(input)

    f = Sndfile(output_file, 'w')

    data = array(output)
    f.write_frames(data)






def practice(file_name):
    model = createModel()
    model.enableInference({'predictedField': 'amp'})
    #logger.info("field info: " + str([x for x in model.getFieldInfo()]))
    #  logger.info("inference type: " + model.getInferenceType())

    metricsManager = MetricsManager(_METRIC_SPECS, model.getFieldInfo(),
                                      model.getInferenceType())

    logger.info("learning from file %s", file_name)
    f = Sndfile(file_name, 'r')

    #data, fs, enc = aiffread('../input/test.aiff')
    # Sndfile instances can be queried for the audio file meta-data

    predicted = []
    raw = []
    prediction = 0.0
    count = 0
    #frames = data.read_frames(1000)

    total_batches = 100

    actual = f.read_frames(1000, dtype=numpy.float64)
    batch_count = 1
    while batch_count <= total_batches:
        for i, record2 in actual:
            #print record

            count += 1
            if count % 100 == 0:
                logger.info("frame %s, Prediction: %s [%s]", count, str(prediction), record2)

            modelInput = {'amp': record2}
            #logger.info("Passing in %s", modelInput["audio"])

            #this is where the magic happens:
            result = model.run(modelInput)

            prediction = result.inferences['prediction']

            #result.metrics = metricsManager.update(result)


            raw.append(record2)
            predicted.append(prediction)
            #numpy.append(predicted, prediction)
            #logger.info(result.metrics)
            #logger.info(metricsManager.getMetrics())

            #predicted = numpy.float64(0)
            ##play one channel the predictions and the other the actuals
            #sounds = numpy.array([record, record])
            #scikits.audiolab.play(sounds)

        logger.info("Reading batch %s", batch_count)

        actual = f.read_frames(1000, dtype=numpy.float64)
        batch_count += 1


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

    logger.info("saving to temp file")

    f = Sndfile('temp.aiff', 'w', format=f.format, channels=1, samplerate=f.samplerate)

    data = array(predicted)
    f.write_frames(data)

    model.save("/Users/az/nupic-env/Bill/brains/test")


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)

  practice(sys.argv[1])
  #perform(sys.argv[1], 1000)
