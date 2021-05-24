import numpy as np
import os
import csv
import matplotlib.pyplot as plt

from classes.Rule import Rule
from util.readFile import readCSV
from util.codes import silenceCode, therapist, therapistCode, userCode

# Lowering of pitch in speech signal --> Back-channel


class Rule1(Rule):

    def analyseRule(self):
        # Generate XXX_ProsodyACF.csv
        self.__generatePROSODYACF()

        # Generate XXX_pitch.csv
        self.__generatePitch()

        # Generate Graph
        self.__generateGraphPitch()

    def __generatePROSODYACF(self):
        command = "SMILExtract -C myconfig/prosody/prosodyAcf.conf -I ./files/corpus/##ID##_P/##ID##_AUDIO.wav -csvoutput ./files/results/##ID##_P/rule1/##OUTPUTCSV##"

        command = command.replace("##ID##", str(self.audio))
        command = command.replace(
            "##OUTPUTCSV##", str(self.audio) + "_ProsodyPitch.csv")

        os.system(command)

    def __generatePitch(self):
        prosodyPitch = readCSV('./files/results/' + str(self.audio) +
                               '_P/rule1/' + str(self.audio) + "_ProsodyPitch.csv", ";")

        rowsTranscript = self.transcript.shape[0]

        frameTime = prosodyPitch[:, 1]
        frameTime = frameTime.astype(np.float)

        loudness = prosodyPitch[:, 4]
        loudness = loudness.astype(np.float)

        start_time = self.transcript[:, 0]
        start_time = start_time.astype(np.float)

        stop_time = self.transcript[:, 1]
        stop_time = stop_time.astype(np.float)

        speakers = self.transcript[:, 2]

        path = './files/results/' + \
            str(self.audio) + '_P/rule1/' + str(self.audio) + '_pitch.csv'

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)

            # The speaker = {0 = Silence, 1 = Ellie, 2 = Participant}
            writer.writerow(["frameTime", "pitch", "speaker"])

            currentSpeaker = 0

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

    def __generateGraphPitch(self):
        path = './files/results/' + \
            str(self.audio) + '_P/rule1/' + str(self.audio) + '_pitch.csv'

        file = readCSV(path, ",")

        rowsFile = file.shape[0]

        pitch = file[:, 1]
        pitch = pitch.astype(np.float)
        speaker = file[:, 2]
        speaker = speaker.astype(np.int)

        x = file[:, 0]
        x = x.astype(np.float)

        # Create the values for the line of Ellie and the Participant
        ellie = np.zeros(rowsFile)
        participant = np.zeros(rowsFile)

        for pos in np.arange(rowsFile):
            if (speaker[pos] == 1):  # 1 == Ellie
                ellie[pos] = pitch[pos]
            elif (speaker[pos] == 2):  # 2 == Participant
                participant[pos] = pitch[pos]

        plt.plot(x, ellie, label='Ellie')
        plt.plot(x, participant, label='Participant')

        plt.title('Pitch in ' + str(self.audio) + '_P')
        plt.xlabel('frameTime')
        plt.ylabel('Pitch')
        plt.legend(loc='upper right')

        plt.savefig('./files/results/' + str(self.audio) +
                    '_P/rule1/' + str(self.audio) + '_pitch.png')
        # plt.show()
        # plt.close()
        plt.clf()
