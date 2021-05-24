import numpy as np
import os
import csv
import matplotlib.pyplot as plt

from classes.Rule import Rule
from util.readFile import readCSV
from util.codes import silenceCode, therapist, therapistCode, userCode

# Raised loudness in speech signal --> Back-channel


class Rule2(Rule):

    def analyseRule(self):
        # Generate XXX_PROSODY.csv
        self.__generatePROSODY()

        # Generate XXX_loudness.csv
        self.__generateLoudness()

        # Generate the Graph showing the loudness of Ellie (the therapist) and the participant
        self.__generateGraphLoudness()

    def __generatePROSODY(self):
        command = "SMILExtract -C myconfig/prosody/prosodyShs.conf -I ./files/corpus/##ID##_P/##ID##_AUDIO.wav -csvoutput ./files/results/##ID##_P/rule2/##OUTPUTCSV##"

        command = command.replace("##ID##", str(self.audio))
        command = command.replace(
            "##OUTPUTCSV##", str(self.audio) + "_ProsodyLoudness.csv")

        os.system(command)

    def __generateLoudness(self):

        path = './files/results/' + \
            str(self.audio) + '_P/rule2/' + \
            str(self.audio) + "_ProsodyLoudness.csv"
        prosody = readCSV(path, ";")

        rowsTranscript = self.transcript.shape[0]

        frameTime = prosody[:, 1]
        frameTime = frameTime.astype(np.float)

        loudness = prosody[:, 4]
        loudness = loudness.astype(np.float)

        start_time = self.transcript[:, 0]
        start_time = start_time.astype(np.float)

        stop_time = self.transcript[:, 1]
        stop_time = stop_time.astype(np.float)

        speakers = self.transcript[:, 2]

        path = './files/results/' + \
            str(self.audio) + '_P/rule2/' + \
            str(self.audio) + '_loudness.csv'

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)

            # The speaker = {0 = Silence, 1 = Ellie, 2 = Participant}
            writer.writerow(["frameTime", "loudness", "speaker"])

            currentSpeaker = silenceCode

            rowsProsody = frameTime.shape[0]
            rowsTranscript = start_time.shape[0]

            # Write the rows before start_time[0]
            for currentRowProsody in np.arange(rowsProsody):
                if (frameTime[currentRowProsody] > start_time[0]):
                    break

                writer.writerow([frameTime[currentRowProsody],
                                 loudness[currentRowProsody], currentSpeaker])

            oldB = 0.0
            for pos in np.arange(rowsTranscript):
                a = start_time[pos]
                b = stop_time[pos]

                # Silence moment
                if(oldB < frameTime[currentRowProsody] < a):

                    currentSpeaker = silenceCode

                    writer.writerow([frameTime[currentRowProsody],
                                     loudness[currentRowProsody], currentSpeaker])
                    currentRowProsody += 1

                    while(oldB < frameTime[currentRowProsody] < a):
                        writer.writerow(
                            [frameTime[currentRowProsody], loudness[currentRowProsody], currentSpeaker])
                        currentRowProsody += 1

                # Detect currentSpeaker
                if (speakers[pos] == therapist):
                    currentSpeaker = therapistCode
                else:
                    currentSpeaker = userCode

                while (a <= frameTime[currentRowProsody] <= b):

                    writer.writerow([frameTime[currentRowProsody],
                                     loudness[currentRowProsody], currentSpeaker])
                    currentRowProsody += 1

                oldB = b

    def __generateGraphLoudness(self):
        path = './files/results/' + \
            str(self.audio) + '_P/rule2/' + \
            str(self.audio) + '_loudness.csv'

        file = readCSV(path, ",")

        rowsFile = file.shape[0]

        loudness = file[:, 1]
        loudness = loudness.astype(np.float)
        speaker = file[:, 2]
        speaker = speaker.astype(np.int)

        x = file[:, 0]
        x = x.astype(np.float)

        # Create the values for the line of Ellie and the Participant
        ellie = np.zeros(rowsFile)
        participant = np.zeros(rowsFile)

        for pos in np.arange(rowsFile):
            if (speaker[pos] == therapistCode):
                ellie[pos] = loudness[pos]
            elif (speaker[pos] == userCode):
                participant[pos] = loudness[pos]

        plt.plot(x, ellie, label='Ellie')
        plt.plot(x, participant, label='Participant')

        plt.title('Loudness of ' + str(self.audio) + '_P')
        plt.xlabel('frameTime')
        plt.ylabel('Loudness')
        plt.legend(loc='upper right')

        pathFig = './files/results/' + \
            str(self.audio) + '_P/rule2/' + \
            str(self.audio) + '_loudness.png'
        plt.savefig(pathFig)
        plt.clf()
        # plt.show()
        # plt.close()
