import numpy as np
import csv

from classes.rules.Rule import Rule
from util.readFile import readCSV
from util.codes import silenceCode, therapist, therapistCode, participant, participantCode

# A region of pitch less than the 26th-percentile pitch level and continuing for at least 110 milliseconds --> Back-channel


class Rule4(Rule):

    def analyseRule(self):
        # It needs the file XXX_pitch.csv generated in Rule 1
        path = './files/results/' + \
            str(self.audio) + '_P/rule1/' + str(self.audio) + "_pitch.csv"
        self.__file = readCSV(path, ",")

        ab = self.__file[:, [0, 1]].astype(np.float)
        c = self.__file[:, 2].astype(np.int).reshape(-1, 1)
        self.__newfile = np.concatenate((ab, c), axis=1)

        self.__frameTime = self.__newfile[:, 0]

        self.__pitch = self.__newfile[:, 1]

        self.__speaker = self.__newfile[:, 2]

        # Compute the Xth-percentile pitch level
        self.__percentile = 26
        self.__x = self.__computePercentilePitchLevel()

        # Generate XXX_region26Pitch.csv
        self.__generateRegionPercentilePitch()

        print("Rule 4 finished...")

    def __computePercentilePitchLevel(self):

        # Get the values of speaker = participantCode
        values = self.__newfile[np.where(
            self.__newfile[:, 2] == participantCode)]

        totalUniqueValues = np.unique(values)
        totalUniqueValuesSorted = np.sort(totalUniqueValues)

        position = (
            totalUniqueValuesSorted.shape[0] * self.__percentile) / 100.0

        result = totalUniqueValuesSorted[int(position)]
        # print("The " + str(self.__percentile) + "th-percentile pitch level is " +
        #      str(result))

        return result

    def __generateRegionPercentilePitch(self):
        self.__minTime = 0.110

        rowsPitch = self.__frameTime.shape[0]

        startConversation = self.transcript[0, 0].astype(np.float)
        rowsTranscript = self.transcript.shape[0]

        start_time = self.transcript[:, 0]
        start_time = start_time.astype(np.float)

        stop_time = self.transcript[:, 1]
        stop_time = stop_time.astype(np.float)

        self.__speakerTranscript = self.transcript[:, 2]
        self.__speakerTranscript[self.__speakerTranscript ==
                                 therapist] = therapistCode
        self.__speakerTranscript[self.__speakerTranscript ==
                                 participant] = participantCode
        self.__speakerTranscript = self.__speakerTranscript.astype(np.int)

        values = self.transcript[:, 3]

        path = './files/results/' + \
            str(self.audio) + '_P/rule4/' + str(self.audio) + \
            '_region' + str(self.__percentile) + 'Pitch.csv'

        with open(path, 'w', newline='') as file:
            self.__writer = csv.writer(file)

            self.__writer.writerow(
                ["recommended_time", "start_time", "answer"])

            self.__totalTime = 0.0
            self.__lastDifferentSpeaker = silenceCode
            self.__oldSpeaker = silenceCode

            # Avoid checking the time before the first sentence
            for self.__currentRowPitch in np.arange(rowsPitch):
                if(self.__frameTime[self.__currentRowPitch] > startConversation):
                    break

            for pos in np.arange(0, rowsTranscript - 1):
                self.__a = start_time[pos]
                self.__b = stop_time[pos]

                # Jump the silences and the sentences of the therapist
                while(self.__currentRowPitch < rowsPitch and self.__speaker[self.__currentRowPitch] in (silenceCode, therapistCode)):
                    self.__currentRowPitch += 1

                # Only when the participant is talking
                while ((self.__currentRowPitch < rowsPitch)
                       and self.__speaker[self.__currentRowPitch] == participantCode
                       and (self.__a <= self.__frameTime[self.__currentRowPitch] <= self.__b)):

                    # Get the next sentence of the therapist
                    self.__sentence = self.__getTherapistAnswer(values, pos)
                    #self.__sentence = values[pos + 1]

                    # Get nextSpeaker
                    self.__nextSpeaker = self.__speakerTranscript[pos + 1]

                    # Get the next 'a' time
                    self.__nextA = start_time[pos + 1]

                    self.__checkWriteRowPitch()

    def __checkWriteRowPitch(self):

        self.__actualSpeaker = self.__speaker[self.__currentRowPitch]

        if(self.__oldSpeaker != self.__actualSpeaker):
            self.__lastDifferentSpeaker = self.__oldSpeaker

        # Check if the pitch is less than the Xth-percentile pitch level
        # and the last element and the actual element being checked have the same speaker
        # and the speaker is NOT the therapist
        # and the last different speaker was NOT the therapist
        if (self.__nextSpeaker != self.__actualSpeaker
            and (self.__pitch[self.__currentRowPitch] < self.__x)
            and (self.__oldSpeaker == self.__actualSpeaker)
            and (self.__actualSpeaker != therapistCode)
                and (self.__lastDifferentSpeaker != therapistCode)):

            self.__totalTime += self.__frameTime[self.__currentRowPitch]

            if (self.__totalTime >= self.__minTime):
                self.__writer.writerow(
                    [self.__frameTime[self.__currentRowPitch], self.__nextA, self.__sentence])
        else:
            self.__totalTime = 0.0

        self.__oldSpeaker = self.__actualSpeaker
        self.__currentRowPitch += 1

    def __getTherapistAnswer(self, values, idx):

        for x in np.arange(idx, self.transcript.shape[0]):

            if self.__speakerTranscript[x] == therapistCode:
                return values[x]

        return "null"
