import numpy as np
import csv

from util.readFile import readCSV
from util.codes import startCorpus, endCorpus


class GroupConfusionMatrix:

    def group(self):

        path = './files/results/ConfusionMatrix.csv'

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["id", "TN", "FP", "FN", "TP", "n"])

            for c in np.arange(startCorpus, endCorpus + 1):

                if (c != 316):

                    path = './files/results/' + \
                        str(c) + '_P/confusionMatrix.csv'

                    file = readCSV(path, ",")
                    row = file[0]

                    writer.writerow(
                        [c, row[0], row[1], row[2], row[3], row[4]])
