import numpy as np

from util.readFile import readCSV
from util.codes import lNoAct, lBackChannel, lQuestion


class CheckPrediction:

    def __init__(self, audio):
        self.audio = audio

        # Create range with a variable
        self.range = 1.5

        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

        self.actualYes = 0
        self.actualNo = 0

        self.predictedNo = 0
        self.predictedYes = 0

    def checkPrediction(self):
        pathPrediction = './files/results/' + \
            str(self.audio) + '_P/prediction.csv'

        prediction = readCSV(pathPrediction, ",")
        fTPrediction = prediction[:, 0].astype(np.float)
        labelPredict = prediction[:, 1].astype(np.int)

        pathSolution = './files/results/' + \
            str(self.audio) + '_P/solution.csv'

        solution = readCSV(pathSolution, ",")
        fTSolution = solution[:, 0].astype(np.float)
        labelSolution = solution[:, 1].astype(np.float)

        # Get Total Backchannel
        self.__total = labelSolution[labelSolution == lBackChannel].shape[0]

        # Start Checking Prediction
        for pos, label in enumerate(labelPredict):

            if label == lBackChannel:
                # Check lBackChannel

                a = fTPrediction[pos] <= fTSolution
                b = fTSolution <= (fTPrediction[pos] + self.range)

                interval = labelSolution[a & b]

                if any([i == lBackChannel for i in interval]):
                    self.TP += 1
                else:
                    self.FP += 1

            elif label == lQuestion:
                continue

            elif label == lNoAct:
                # Check lNoAct
                a = fTPrediction[pos] <= fTSolution
                b = fTSolution <= (fTPrediction[pos] + self.range)

                interval = labelSolution[a & b]

                if all([i != lBackChannel for i in interval]):
                    self.TN += 1
                

        for pos, label in enumerate(labelSolution):

            if label == lBackChannel:
                # Check lBackChannel

                a = (fTSolution[pos] - self.range) <= fTPrediction
                b = fTPrediction <= fTSolution[pos]

                interval = labelPredict[a & b]

                if all([i != lBackChannel for i in interval]):
                    self.FN += 1

            elif label == lQuestion:
                continue

        print("Total = " + str(self.__total))
        print("TP = " + str(self.TP))
        print("FP = " + str(self.FP))
        print("TN = " + str(self.TN))
        print("FN = " + str(self.FN))

    def __computeAccuracy(self):
        self.__accuracy = (self.TP + self.TN) / self.__total

    def __computeMisclassificationRate(self):
        self.__misClasRate = (self.FP + self.FN) / self.__total

    def __computeTP_rate(self):
        self.__TPrate = self.TP / self.actualYes

    def __computeFP_rate(self):
        self.__FPrate = self.FP / self.actualNo

    def __computeTN_rate(self):
        self.__TNrate = self.TN / self.actualNo

    def __computePrecision(self):
        self.__precision = self.TP / self.predictedYes

    def __computePrevalence(self):
        self.__prevalence = self.actualYes / self.__total

    # recall
