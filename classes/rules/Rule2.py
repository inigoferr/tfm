import numpy as np
import os
import csv
import matplotlib.pyplot as plt

from classes.rules.Rule import Rule
from util.readFile import readCSV
from util.codes import silenceCode, therapist, therapistCode, participant, participantCode

# Raised loudness in speech signal --> Back-channel


class Rule2(Rule):

    def analyseRule(self):
        self.__start_time = self.transcript[:, 0]
        self.__start_time = self.__start_time.astype(np.float)

        self.__stop_time = self.transcript[:, 1]
        self.__stop_time = self.__stop_time.astype(np.float)

        self.__speakers = self.transcript[:, 2]

        self.__speakers[self.__speakers == therapist] = therapistCode
        self.__speakers[self.__speakers == participant] = participantCode
        self.__speakers = self.__speakers.astype(np.int)
        # Generate XXX_PROSODY.csv
        self.__generatePROSODY()

        # Generate XXX_loudness.csv
        self.__generateLoudness()

        # Generate dictionaryAnswers.csv
        self.__generateAnswersPyschologist()

        # Generate the Graph showing the loudness of Ellie (the therapist) and the participant
        self.__generateGraphLoudness()

        print("Rule 2 finished...")

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

        path = './files/results/' + \
            str(self.audio) + '_P/rule2/' + \
            str(self.audio) + '_loudness.csv'

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)

            # The speaker = {0 = Silence, 1 = Ellie, 2 = Participant}
            writer.writerow(["frameTime", "loudness", "speaker"])

            currentSpeaker = silenceCode

            rowsProsody = frameTime.shape[0]
            rowsTranscript = self.__start_time.shape[0]

            # Write the rows before start_time[0]
            for currentRowProsody in np.arange(rowsProsody):
                if (frameTime[currentRowProsody] > self.__start_time[0]):
                    break

                writer.writerow([frameTime[currentRowProsody],
                                 loudness[currentRowProsody], currentSpeaker])

            oldB = 0.0
            for pos in np.arange(rowsTranscript):
                a = self.__start_time[pos]
                b = self.__stop_time[pos]

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

                currentSpeaker = self.__speakers[pos]

                while (a <= frameTime[currentRowProsody] <= b):

                    writer.writerow([frameTime[currentRowProsody],
                                     loudness[currentRowProsody], currentSpeaker])
                    currentRowProsody += 1

                oldB = b

    def __generateAnswersPyschologist(self):
        path = './files/results/' + \
            str(self.audio) + '_P/rule2/' + str(self.audio) + '_loudness.csv'

        file = readCSV(path, ",")

        rowsLoudness = file.shape[0]
        rowsTranscript = self.transcript.shape[0]
        startConversation = self.transcript[0, 0].astype(np.float)

        values = self.transcript[:, 3]

        frameTime = file[:, 0].astype(np.float)
        loudness = file[:, 1].astype(np.float)
        speakerLoudness = file[:, 2].astype(np.int)

        diff = frameTime[1] - frameTime[0]

        range = diff * 5

        pathDictionary = './files/results/' + \
            str(self.audio) + '_P/rule2/dictionaryAnswers.csv'

        with open(pathDictionary, 'w', newline='') as fileDictionary:
            writer = csv.writer(fileDictionary)
            writer.writerow(["totalTime", "recommended_time",
                             "start_time", "speaker", "answer"])

            totalTime = 0.0

            # Avoid checking the time before the first sentence
            for currentRowLoudness in np.arange(rowsLoudness):
                if(frameTime[currentRowLoudness] > startConversation):
                    break

            previousLoudness = 0.0
            for pos in np.arange(0, rowsTranscript - 1):
                a = self.__start_time[pos]
                b = self.__stop_time[pos]

                # Jump the silences and the sentences of the therapist
                while(currentRowLoudness < rowsLoudness and speakerLoudness[currentRowLoudness] in (silenceCode, therapistCode)):
                    currentRowLoudness += 1

                # Only when the user is talking
                while ((currentRowLoudness < rowsLoudness)
                       and speakerLoudness[currentRowLoudness] == participantCode
                       and (a <= frameTime[currentRowLoudness] <= b)):

                    actualLoudness = loudness[currentRowLoudness]

                    if (previousLoudness <= actualLoudness):
                        totalTime += diff
                    else:
                        totalTime = 0.0

                    # Get the next 'a' time
                    nextA = self.__start_time[pos + 1]

                    if (totalTime >= range):
                        writer.writerow(
                            [totalTime, frameTime[currentRowLoudness], nextA, self.__speakers[pos + 1], values[pos + 1]])

                    currentRowLoudness += 1
                    previousLoudness = actualLoudness

                totalTime = 0.0

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
            elif (speaker[pos] == participantCode):
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
