import numpy as np
import csv

from classes.Rule import Rule
from util.codes import therapist, therapistCode, participant, participantCode

# After at least 700 milliseconds of speech --> Back-channel


class Rule5(Rule):

    def __init__(self, audio, transcript):
        super().__init__(audio, transcript)

        # List with all the answers of the Pyschologist
        self.__answers = []

    def analyseRule(self):

        self.__start_time = self.transcript[:, 0]
        self.__start_time = self.__start_time.astype(np.float)

        self.__stop_time = self.transcript[:, 1]
        self.__stop_time = self.__stop_time.astype(np.float)

        self.__speakers = self.transcript[:, 2]
        self.__speakers[self.__speakers == therapist] = therapistCode
        self.__speakers[self.__speakers == participant] = participantCode
        self.__speakers = self.__speakers.astype(np.int)

        self.__values = self.transcript[:, 3]

        self.__previousSpeaker = self.__speakers[0]

        self.__totalTime = 0.0

        for pos, elem in enumerate(self.__values):

            self.__actualSpeaker = self.__speakers[pos]
            self.__answer = elem.strip()

            if (self.__actualSpeaker == participantCode):
                self.__totalTime += (self.__stop_time[pos] -
                                     self.__start_time[pos])

                # Greater than 700ms
                if (self.__totalTime >= 0.700):
                    self.__sT = self.__start_time[pos]
                    self.__saveAnswer()

            else:  # Therapist
                # Save Row if totalTime greater than 0.700
                if (self.__previousSpeaker == therapistCode and self.__totalTime >= 0.700):
                    self.__saveAnswer()

                self.__totalTime = 0.0

            self.__previousSpeaker = self.__actualSpeaker

        # Create File with answers of the Pyschologist
        self.__answers = np.unique(np.array(self.__answers), axis=0)

        path = './files/results/' + \
            str(self.audio) + '_P/rule5/dictionaryAnswersPyschologist.csv'
        file = open(path, "w")

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["totalTime", "speaker", "start_time", "answer"])

            for a in self.__answers:
                writer.writerow([a[0], a[1], a[2], a[3]])

        print('Rule 5 finished...')

    def __saveAnswer(self):
        self.__answers.append(
            [self.__totalTime, self.__actualSpeaker, self.__sT, self.__answer])
