from classes.evaluateRules.EvalRule import EvalRule


class EvalRule2(EvalRule):

    def __init__(self, frameSize, t):
        self.frameSize = frameSize
        self.t = t

    def evaluateRule(self, actualLoudness, previousLoudness, loudnessIncrease):
        # Rule 2
        if actualLoudness >= previousLoudness:
            loudnessIncrease += self.frameSize

            if (loudnessIncrease >= self.t):  # Back-Channel
                # We think, we should do back-channel
                return True, loudnessIncrease

            return False, loudnessIncrease
        else:
            loudnessIncrease = 0.0
            return False, loudnessIncrease
