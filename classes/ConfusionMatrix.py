import numpy as np

from util.readFile import readCSV


class ConfusionMatrix:

    def __init__(self, audio):
        self.audio = audio

    def showConfusionMatrix(self):

        path = './files/results/' + \
            str(self.audio) + '_P/confusionMatrix.csv'

        file = readCSV(path, ",")

        self.TN = file[0, 0].astype(np.int)
        self.FP = file[0, 1].astype(np.int)
        self.FN = file[0, 2].astype(np.int)
        self.TP = file[0, 3].astype(np.int)
        self.total = file[0, 4].astype(np.int)


"""
        print("Total = " + str(self.total))
        print("TP = " + str(self.TP))
        print("FP = " + str(self.FP))
        print("TN = " + str(self.TN))
        print("FN = " + str(self.FN))
"""
