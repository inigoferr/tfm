import numpy as np
import csv

from classes.rules.Rule import Rule
from util.readFile import readCSV
"""
Not output back-channel feedback within the preceding 800 milliseconds --> Back-channel
"""


class Rule7(Rule):

    def analyseRule(self):

        # Preceding time
        time = 0.800

        # Read the XXX_region26Pitch.csv
        file = readCSV('./files/results/' + str(self.audio) +
                       '_P/rule4/' + str(self.audio) + "_region26Pitch.csv", ",")

        start_time = file[:, 1]
        start_time = start_time.astype(np.float)
        start_time = np.unique(start_time)

        path = './files/results/' + \
            str(self.audio) + '_P/rule7/' + \
            str(self.audio) + "_feedbackPreceding.csv"

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["start_time1", "start_time2"])

            empty = True
            for pos in np.arange(0, start_time.shape[0] - 1):

                # Check if the next feedback was done before or not the correct time
                if(start_time[pos] + time >= start_time[pos + 1]):
                    writer.writerow(
                        [start_time[pos], start_time[pos + 1]])
                    empty = False

            if empty:
                writer.writerow(["None", "None"])

        print("Rule 7 finished...")
