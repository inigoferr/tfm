import numpy as np

from util.readFile import readCSV
from util.codes import therapist


class Evaluation:

    def __init__(self, audio, transcript):
        self.audio = audio
        self.transcript = transcript

    def evaluate(self):

        # Filter XXX_Dictionary.csv
        self.__filterAnswers()

        # Create range with a variable
        self.__range = 1.0

        # Create confusion matrix
        self.__generateConfusionMatrix()

    def __filterAnswers(self):

        # Get all the interventions of the Therapists
        matrix = self.transcript[self.transcript[:, 2] == therapist]

        answers = matrix[:, 3].astype('U')

        words = ("hi", "what", "why", "how", "who", "when",
                 "where", "what's", "do", "did", "can", "could", "have", "had", "are", "is")

        results = []

        for x in answers:
            if (not x.strip().startswith(words)):
                results.append(x)

        self.__numBackChannels = len(results)

    def __generateConfusionMatrix(self):

        self.__total = self.__numBackChannels

        self.__TP = 0
        self.__FP = 0
        self.__FN = 0
        self.__TN = 0

        self.__actualYes = 0
        self.__actualNo = 0

        self.__predictedNo = 0
        self.__predictedYes = 0

    def __computeAccuracy(self):
        self.__accuracy = (self.__TP + self.__TN) / self.__total

    def __computeMisclassificationRate(self):
        self.__misClasRate = (self.__FP + self.__FN) / self.__total

    def __computeTP_rate(self):
        self.__TPrate = self.__TP / self.__actualYes

    def __computeFP_rate(self):
        self.__FPrate = self.__FP / self.__actualNo

    def __computeTN_rate(self):
        self.__TNrate = self.__TN / self.__actualNo

    def __computePrecision(self):
        self.__precision = self.__TP / self.__predictedYes

    def __computePrevalence(self):
        self.__prevalence = self.__actualYes / self.__total
