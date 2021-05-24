import numpy as np
import csv

from classes.Rule import Rule
from util.codes import therapist, therapistCode, user, userCode, start_time, stop_time, speaker, value

# After at least 700 milliseconds of speech --> Back-channel


class Rule5(Rule):

    def __init__(self, audio, transcript):
        super().__init__(audio, transcript)

        # Columns for the matrix 'speech'
        self.__posDuration = 0
        self.__posSpeaker = 1
        self.__posNumLines = 2
        self.__posFirstLine = 3

        # List with all the answers of the Pyschologist
        self.__answers = []

    def analyseRule(self):
        rows = self.transcript.shape[0]

        # Compute the duration of speech

        # Structure of one row of the self.transcript 'speech' ['durationOfTheSpeech' 'codeOfTheSpeaker' 'numLinesInTRANSCRIPT' 'firstLineInTRANSCRIPT' ]
        self.__speech = np.zeros((rows, 4))

        self.__accumulatedDuration = 0
        self.__previousSpeaker = self.transcript[0, speaker]
        self.__numLines = 0
        self.__firstLine = 2

        for x in np.arange(0, rows):
            actualSpeaker = self.transcript[x, speaker]

            # Only interested on user's speech, not therapist's
            if (actualSpeaker == user):
                # The speaker talking is different from the previous one
                if (actualSpeaker != self.__previousSpeaker):
                    # Save the speech of the previousSpeaker
                    self.__row = x - 1
                    self.__saveSpeechInRow()

                    # Reset values
                    self.__accumulatedDuration = (float(self.transcript[x, stop_time]) -
                                                  float(self.transcript[x, start_time]))
                    self.__numLines = 1
                    self.__firstLine = x + 2
                else:  # The speaker talking is the same from the previous sentence
                    self.__accumulatedDuration += (float(self.transcript[x, stop_time]) -
                                                   float(self.transcript[x, start_time]))
                    self.__numLines += 1

            self.__previousSpeaker = actualSpeaker

        # Save the last sentence if necessary
        self.__row = rows - 1
        self.__saveSpeechInRow()

        # Get rows with duration more than 700 milliseconds
        maskNoZeros = (self.__speech[:, self.__posDuration] > 7.0)
        self.__speech = self.__speech[maskNoZeros]

        # Print the answer of the Ellie (Psychologist) after 700 milliseconds of speech
        rows = self.__speech.shape[0]
        for x in np.arange(0, rows):
            self.__firstLine = self.__speech[x, 2]
            self.__numLines = self.__speech[x, 3]
            self.__saveAnswers()

        # Create File with answers of the Pyschologist
        self.__answers = np.unique(np.array(self.__answers))

        path = './files/results/' + \
            str(self.audio) + '_P/rule5/dictionaryAnswersPyschologist.csv'
        file = open(path, "w")

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["answer"])

            for word in self.__answers:
                writer.writerow([word])
        
        print('Rule 5 finished...')

    def __saveAnswers(self):
        self.__answers.append(self.transcript[int(self.__firstLine) +
                                              int(self.__numLines) + 1, value].strip())

    def __saveSpeechInRow(self):
        self.__speech[self.__row,
                      self.__posDuration] = self.__accumulatedDuration
        self.__speech[self.__row, self.__posSpeaker] = therapistCode if (
            self.__previousSpeaker == therapist) else userCode
        self.__speech[self.__row, self.__posNumLines] = self.__numLines
        self.__speech[self.__row, self.__posFirstLine] = self.__firstLine
