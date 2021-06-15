import numpy as np
import os
import csv
import matplotlib.pyplot as plt

from classes.rules.Rule import Rule
from util.readFile import readCSV
from util.codes import silenceCode, therapist, therapistCode, participant, participantCode

# Lowering of pitch in speech signal --> Back-channel


class Rule1(Rule):

    def analyseRule(self):
        self.__start_time = self.transcript[:, 0]
        self.__start_time = self.__start_time.astype(np.float)

        self.__stop_time = self.transcript[:, 1]
        self.__stop_time = self.__stop_time.astype(np.float)

        self.__speakers = self.transcript[:, 2]

        self.__speakers[self.__speakers == therapist] = therapistCode
        self.__speakers[self.__speakers == participant] = participantCode
        self.__speakers = self.__speakers.astype(np.int)

        # Generate XXX_ProsodyACF.csv
        self.__generatePROSODYACF()

        # Generate XXX_pitch.csv
        self.__generatePitch()

        # Generate XXX_dictionaryAnswersPyschologist.csv
        self.__generateAnswersPyschologist()

        # Generate Graph
        self.__generateGraphPitch()

        print("Rule 1 finished...")

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

        pitch = prosodyPitch[:, 3]
        pitch = pitch.astype(np.float)

        path = './files/results/' + \
            str(self.audio) + '_P/rule1/' + str(self.audio) + '_pitch.csv'

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)

            # The speaker = {0 = Silence, 1 = Ellie, 2 = Participant}
            writer.writerow(["frameTime", "pitch", "speaker"])

            currentSpeaker = 0

            rowsProsody = frameTime.shape[0]
            rowsTranscript = self.__start_time.shape[0]

            # Write the rows before self.__start_time[0]
            for currentRowProsody in np.arange(rowsProsody):
                if (frameTime[currentRowProsody] > self.__start_time[0]):
                    break

                writer.writerow([frameTime[currentRowProsody],
                                 pitch[currentRowProsody], currentSpeaker])

            oldB = 0.0
            for pos in np.arange(rowsTranscript):
                a = self.__start_time[pos]
                b = self.__stop_time[pos]

                # Silence moment
                if(oldB < frameTime[currentRowProsody] < a):

                    currentSpeaker = silenceCode

                    writer.writerow([frameTime[currentRowProsody],
                                     pitch[currentRowProsody], currentSpeaker])
                    currentRowProsody += 1

                    while(oldB < frameTime[currentRowProsody] < a):
                        writer.writerow(
                            [frameTime[currentRowProsody], pitch[currentRowProsody], currentSpeaker])
                        currentRowProsody += 1

                currentSpeaker = self.__speakers[pos]

                while (a <= frameTime[currentRowProsody] <= b):

                    writer.writerow([frameTime[currentRowProsody],
                                     pitch[currentRowProsody], currentSpeaker])
                    currentRowProsody += 1

                oldB = b

    def __generateAnswersPyschologist(self):
        path = './files/results/' + \
            str(self.audio) + '_P/rule1/' + str(self.audio) + '_pitch.csv'

        file = readCSV(path, ",")

        rowsPitch = file.shape[0]
        rowsTranscript = self.transcript.shape[0]
        startConversation = self.transcript[0, 0].astype(np.float)

        values = self.transcript[:, 3]

        frameTime = file[:, 0].astype(np.float)
        pitch = file[:, 1].astype(np.float)
        speakerPitch = file[:, 2].astype(np.int)

        diff = frameTime[1] - frameTime[0]

        range = diff * 5

        pathDictionary = './files/results/' + \
            str(self.audio) + '_P/rule1/dictionaryAnswers.csv'

        with open(pathDictionary, 'w', newline='') as fileDictionary:
            writer = csv.writer(fileDictionary)
            writer.writerow(["totalTime", "recommended_time",
                             "start_time", "speaker", "answer"])

            totalTime = 0.0

            # Avoid checking the time before the first sentence
            for currentRowPitch in np.arange(rowsPitch):
                if(frameTime[currentRowPitch] > startConversation):
                    break

            previousPitch = 0.0
            for pos in np.arange(0, rowsTranscript - 1):
                a = self.__start_time[pos]
                b = self.__stop_time[pos]

                # Jump the silences and the sentences of the therapist
                while(currentRowPitch < rowsPitch and speakerPitch[currentRowPitch] in (silenceCode, therapistCode)):
                    currentRowPitch += 1

                # Only when the participant is talking
                while ((currentRowPitch < rowsPitch)
                       and speakerPitch[currentRowPitch] == participantCode
                       and (a <= frameTime[currentRowPitch] <= b)):

                    actualPitch = pitch[currentRowPitch]

                    if (previousPitch >= actualPitch):
                        totalTime += diff
                    else:
                        totalTime = 0.0

                    # Get the next 'a' time
                    nextA = self.__start_time[pos + 1]

                    if (totalTime >= range):
                        writer.writerow(
                            [totalTime, frameTime[currentRowPitch], nextA, self.__speakers[pos + 1], values[pos + 1]])

                    currentRowPitch += 1
                    previousPitch = actualPitch

                totalTime = 0.0

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
