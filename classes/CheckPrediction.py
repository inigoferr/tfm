import numpy as np
import csv

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
        self.total = labelSolution[labelSolution == lBackChannel].shape[0]

        # List of Bakchannel checked
        cBackchannel = []

        # Start Checking Prediction

        # Get True Positives (TP), False Positives (FP) and True Negatives (TN)
        for pos, label in enumerate(labelPredict):

            if label == lBackChannel:
                # Check lBackChannel

                # Get the range where checking
                a = fTPrediction[pos] <= fTSolution
                b = fTSolution <= (fTPrediction[pos] + self.range)

                interval = labelSolution[a & b]

                startTime = fTSolution[a & b & (labelSolution == lBackChannel)]

                if any([i == lBackChannel for i in interval]):
                    if not startTime[0] in cBackchannel:
                        cBackchannel.append(startTime[0])
                        self.TP += 1
                else:
                    self.FP += 1

            elif label == lQuestion:
                continue

            elif label == lNoAct:
                # Check lNoAct

                # Get the range where checking
                a = fTPrediction[pos] <= fTSolution
                b = fTSolution <= (fTPrediction[pos] + self.range)

                interval = labelSolution[a & b]

                if all([i != lBackChannel for i in interval]):
                    self.TN += 1

        # Get False Negatives (FN)
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

        # Compute Actuals and Predicted
        self.__computeActualsPredicted()

        # Save values into confusionMatrix.csv
        self.__saveValues()

    def __computeActualsPredicted(self):
        self.actualYes = self.TN + self.FP
        self.actualNo = self.FN + self.TP

        self.predictedYes = self.TN + self.FN
        self.predictedNo = self.FP + self.TP

    def __saveValues(self):
        # Copy into confusionMatrix.csv the values of the confusion matrix obtained
        path = './files/results/' + \
            str(self.audio) + '_P/confusionMatrix.csv'

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["TN", "FP", "FN", "TP", "n"])
            writer.writerow([self.TN, self.FP, self.FN, self.TP, self.total])

    def getPrecision(self):
        return self.TP / self.predictedYes

    def getRecall(self):
        return self.TP / (self.TP + self.FN)

    def getF1Score(self):
        return 2 * self.getPrecision() * self.getRecall() / (self.getPrecision() + self.getRecall())

    def getAccuracy(self):
        return (self.TP + self.TN) / self.total

    def getMisclassificationRate(self):
        return (self.FP + self.FN) / self.total

    def getTP_rate(self):
        return self.TP / self.actualYes

    def getFP_rate(self):
        return self.FP / self.actualNo

    def getTN_rate(self):
        return self.TN / self.actualNo

    def getPrevalence(self):
        return self.actualYes / self.total
