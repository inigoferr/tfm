from classes.evaluateRules.EvalRule import EvalRule


class EvalRule1(EvalRule):

    def __init__(self, frameSize, t):
        self.frameSize = frameSize
        self.t = t

    def evaluateRule(self, actualPitch, previousPitch, pitchIncrease):
        if actualPitch <= previousPitch:  # Less or equal to be more flexible
            pitchIncrease += self.frameSize

            if (pitchIncrease >= self.t):  # Back-Channel
                # We think, we should do back-channel
                return True, pitchIncrease
            return False, pitchIncrease
        else:
            pitchIncrease = 0.0
            return False, pitchIncrease
