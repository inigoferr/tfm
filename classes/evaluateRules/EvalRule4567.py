import numpy as np

from classes.evaluateRules.EvalRule import EvalRule


class EvalRule4567(EvalRule):

    def __init__(self, x, frameSize, tPercentile, tSpeech, tSilence, tPreceding):
        self.x = x
        self.frameSize = frameSize
        self.tPercentile = tPercentile
        self.tSpeech = tSpeech
        self.tSilence = tSilence
        self.tPreceding = tPreceding

    def evaluateRule(self, totalUniquePitchValues, actualPitch, timePercentile, timeSpeech, timeSilence, timeNoBackChannel):
        # Get X-thPercentile
        a = np.unique(np.array(totalUniquePitchValues))
        sorted = np.sort(a)

        if sorted.shape[0] > 0:
            position = (sorted.shape[0] * self.x) / 100.0
            result = sorted[int(position)]

            if actualPitch < result:
                timePercentile += self.frameSize
            else:
                timePercentile = 0.0

            if (actualPitch < result
                and timePercentile >= self.tPercentile
                and timeSpeech >= self.tSpeech
                and timeSilence >= self.tSilence
                    and timeNoBackChannel >= self.tPreceding):
                # We think, we should do back-channel
                return True, timePercentile
            else:
                return False, timePercentile
        else:
            return False, timePercentile
