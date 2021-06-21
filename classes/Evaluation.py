from classes.ConfusionMatrix import ConfusionMatrix
import numpy as np
import csv

from classes.evaluateRules.EvalRule4567 import EvalRule4567
from classes.evaluateRules.EvalRule3 import EvalRule3
from classes.evaluateRules.EvalRule2 import EvalRule2
from classes.evaluateRules.EvalRule1 import EvalRule1
from classes.CheckPrediction import CheckPrediction
from classes.Solution import Solution
from util.readFile import readCSV
from util.codes import lNoAct, lBackChannel, lQuestion, participantCode, silenceCode


class Evaluation:

    def __init__(self, audio, transcript):
        self.audio = audio
        self.transcript = transcript

    def evaluate(self):

        # Audio Duration (from the transcript)
        self.__audioDuration = self.transcript[:, 1][-1].astype(np.float)

        # Frame Size
        self.__frameSize = 0.025

        # totalFrames
        self.__totalFrames = np.round(self.__audioDuration /
                                      self.__frameSize).astype(np.int)

        # Generate the solution from the transcript
        solution = Solution(self.audio, self.transcript,
                            self.__totalFrames, self.__frameSize)
        solution.generateSolution()

        # Predict
        self.__predict()

        # Check Prediction
        cP = CheckPrediction(self.audio)
        cP.checkPrediction()

        # Confusion Matrix
        confusionMatrix = ConfusionMatrix(self.audio)
        confusionMatrix.showConfusionMatrix()

    def __predict(self):

        # Init parameters
        self.__initParameters()

        # Get XXX_pitch.csv
        pathPitch = './files/results/' + \
            str(self.audio) + '_P/rule1/' + str(self.audio) + '_pitch.csv'
        filePitch = readCSV(pathPitch, ",")
        pitch = filePitch[:, 1].astype(np.float)
        speaker = filePitch[:, 2].astype(np.int)

        # Get XXX_loudness.csv
        pathLoudness = './files/results/' + \
            str(self.audio) + '_P/rule2/' + str(self.audio) + '_loudness.csv'
        fileLoudness = readCSV(pathLoudness, ",")
        loudness = fileLoudness[:, 1].astype(np.float)

        # Variables
        previousSpeaker = silenceCode
        self.__previousPrediction = lNoAct
        self.__previousPitch = 0.0
        self.__totalUniquePitchValues = []
        self.__previousLoudness = 0.0

        # List with predictions
        self.__predictions = []

        # Initialise EvalRuleX
        self.__rule1 = EvalRule1(self.__frameSize, self.__t)
        self.__rule2 = EvalRule2(self.__frameSize, self.__t)
        self.__rule3 = EvalRule3()
        self.__rule4567 = EvalRule4567(self.__percentile, self.__frameSize,
                                       self.__tPercentile, self.__tSpeech,
                                       self.__tSilence, self.__tPreceding)

        for frame in np.arange(self.__totalFrames - 1):

            # Variable to know if we decide to act or not
            acting = False

            actualSpeaker = speaker[frame]
            self.__actualPitch = pitch[frame]
            self.__actualLoudness = loudness[frame]

            # The speaker changes
            if actualSpeaker != previousSpeaker:
                if actualSpeaker == silenceCode:
                    self.__pitchDecrease = 0.0
                    self.__loudnessIncrease = 0.0
                elif actualSpeaker == participantCode:
                    self.__timePercentile = 0.0
                    self.__timeSpeech = 0.0
                    self.__timeSilence = 0.0
                else:
                    self.__timePercentile = 0.0
                    self.__pitchDecrease = 0.0
                    self.__loudnessIncrease = 0.0
                    self.__timeSilence = 0.0

            # Actual Speaker
            if actualSpeaker == participantCode:  # Participant
                self.__evaluateParticipant(frame)
            elif actualSpeaker == silenceCode:  # Silence
                self.__evaluateSilence(frame)
            else:  # Therapist
                self.__evaluateTherapist(frame)

            previousSpeaker = actualSpeaker
            self.__previousPitch = self.__actualPitch
            self.__previousLoudness = self.__actualLoudness

        # Save prediction into XXX_prediction.csv
        self.__savePrediction()

    def __initParameters(self):
        # Thresholds
        # Pitch and Loudness threshold
        self.__t = self.__frameSize * 5.5
        # Percentile threshold
        self.__tPercentile = 0.11
        # Speech threshold
        self.__tSpeech = 0.7
        # Silence threshold
        self.__tSilence = 0.7
        # Preceding threshold
        self.__tPreceding = 0.8

        # Percentile
        self.__percentile = 26

        # Rules parameters
        self.__pitchDecrease = 0.0
        self.__loudnessIncrease = 0.0
        self.__timePercentile = 0.0
        self.__timeSpeech = 0.0
        self.__timeSilence = 0.0
        self.__timeNoBackChannel = 0.0

    def __evaluateParticipant(self, frame):
        # The user is speaking, so we increase timeSpeech
        self.__timeSpeech += self.__frameSize
        self.__totalUniquePitchValues.append(self.__actualPitch)

        # Apply Rule 1
        r1, self.__pitchDecrease = self.__rule1.evaluateRule(
            self.__actualPitch, self.__previousPitch, self.__pitchDecrease)

        # Apply Rule 2
        r2, self.__loudnessIncrease = self.__rule2.evaluateRule(
            self.__actualLoudness, self.__previousLoudness, self.__loudnessIncrease)

        # Apply Rule 3
        r3 = self.__rule3.evaluateRule()

        acting = r1 or r2 or r3

        if acting:
            if self.__previousPrediction == lBackChannel:
                # Remove last element
                self.__predictions.pop()

                # Add 2 elements; one for modifying the previous frame and other for inserting the actual frame
                self.__predictions.append(
                    [np.round((frame - 1)*self.__frameSize, 3), lNoAct])
                self.__predictions.append(
                    [np.round(frame*self.__frameSize, 3), lBackChannel])
            else:
                self.__predictions.append(
                    [np.round(frame*self.__frameSize, 3), lBackChannel])

            # Update previousPrediction
            self.__previousPrediction = lBackChannel
            self.__timeNoBackChannel = 0.0
        else:

            self.__predictions.append(
                [np.round(frame*self.__frameSize, 3), lNoAct])

            # Update previousPrediction
            self.__previousPrediction = lNoAct

            self.__timeNoBackChannel += self.__frameSize

    def __evaluateSilence(self, frame):
        self.__timeSilence += self.__frameSize
        self.__timeNoBackChannel += self.__frameSize

        # Apply Rule 4,5,6,7
        r4567, self.__timePercentile = self.__rule4567.evaluateRule(
            self.__totalUniquePitchValues, self.__actualPitch, self.__timePercentile, self.__timeSpeech, self.__timeSilence, self.__timeNoBackChannel)

        if r4567:
            if self.__previousPrediction == lBackChannel:
                # Remove last element
                self.__predictions.pop()

                # Add 2 elements; one for modifying the previous frame and other for inserting the actual frame
                self.__predictions.append(
                    [np.round((frame - 1)*self.__frameSize, 3), lNoAct])
                self.__predictions.append(
                    [np.round(frame*self.__frameSize, 3), lBackChannel])
            else:
                self.__predictions.append(
                    [np.round(frame*self.__frameSize, 3), lBackChannel])

                # Update previousPrediction
            self.__previousPrediction = lBackChannel
            self.__timeNoBackChannel = 0.0

        else:
            self.__predictions.append(
                [np.round(frame*self.__frameSize, 3), lNoAct])

            # Update previousPrediction
            self.__previousPrediction = lNoAct

    def __evaluateTherapist(self, frame):
        # Considering when therapist speaks, maybe there is back-channel
        self.__timeNoBackChannel = 0.0
        self.__predictions.append(
            [np.round(frame*self.__frameSize, 3), lNoAct])

        # Update previousPrediction
        self.__previousPrediction = lNoAct

    def __savePrediction(self):
        # Copy into XXX_prediction.csv the predictions
        path = './files/results/' + \
            str(self.audio) + '_P/prediction.csv'

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frameTime", "prediction"])
            writer.writerows(self.__predictions)
