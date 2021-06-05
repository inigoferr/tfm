import numpy as np
import csv

from classes.rules.Rule import Rule
from util.codes import start_time, stop_time, speaker, value

# After 700 milliseconds wait --> Back-channel


class Rule6(Rule):

    def analyseRule(self):
        rows = self.transcript.shape[0]
        # Compute the silences,
        # if silence[z] == 0 --> Means that the speaker of 'z-1' sentence and the 'z' sentence was the same
        # if silence[z] < 0 --> Means that the speakers spoke at the same time or there is an error in the Transcript

        silences = np.zeros(rows - 1)
        before = self.transcript[0, :]
        count = 0
        for x in np.arange(1, rows):
            now = self.transcript[x, :]

            if (now[speaker] != before[speaker]):
                silences[count] = float(
                    now[start_time]) - float(before[stop_time])

            count += 1
            before = now

        # Average Silences
        # averageSilence = np.average(silences)
        # print("Average Silence = " + str(averageSilence))

        # Rows where silence was 700 milliseconds or more
        mask = silences >= 0.7
        self.__positions = np.array(np.where(mask)) + 1
        self.__rows = silences[mask]

        self.__saveSentencesIntoFile()

        print('Rule 6 finished...')

    def __saveSentencesIntoFile(self):
        i = 0

        path = './files/results/' + \
            str(self.audio) + '_P/rule6/dictionarySentences.csv'
        file = open(path, "w")

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["seconds", "line1", "speaker1", "answer1", "start_time", "line2", "speaker2", "answer2"])

            for z in self.__positions:
                for x in z:
                    writer.writerow(
                        [str(np.around(self.__rows[i], 2)),
                            str(x+1),
                            self.transcript[x - 1, speaker],
                            self.transcript[x - 1, value],
                            self.transcript[x, start_time],
                            str(x+2),
                            self.transcript[x, speaker],
                            self.transcript[x, value].strip()
                         ])
                    i += 1
