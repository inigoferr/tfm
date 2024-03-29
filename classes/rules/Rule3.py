import numpy as np
import csv

from classes.rules.Rule import Rule
from util.sylco import sylco
from util.readFile import readCSV
from util.codes import therapist, therapistCode, participant, participantCode, hesitationRepetitionWords, reservedWords
"""
Rule 3: Disfluency in speech signal --> Question to the user
"""


class Rule3(Rule):

    def analyseRule(self):

        self.__start_time = self.transcript[:, 0]
        self.__start_time = self.__start_time.astype(np.float)

        self.__stop_time = self.transcript[:, 1]
        self.__stop_time = self.__stop_time.astype(np.float)

        self.__speaker = self.transcript[:, 2]
        self.__speaker[self.__speaker == therapist] = therapistCode
        self.__speaker[self.__speaker == participant] = participantCode
        self.__speaker = self.__speaker.astype(np.int)

        self.__values = self.transcript[:, 3]

        # self.__computeNumberSyllables()

        # Generate XXX_syllableDuration.csv
        # self.__generateFluencyParameters()

        # Generate XXX_fluencyParameters.csv
        # self.__computeFormulas()

        # Detect disfluencies in the transcript
        self.__detectDisfluenciesTranscript()

        print("Rule 3 finished...")

    def __computeNumberSyllables(self):
        # Initialize totalSyllables
        self.__totalSyllables = np.zeros(
            self.transcript[:, 3].reshape(-1, 1).shape)

        # Obtain nº of Syllables per sentence
        for pos, sentence in enumerate(self.__values):

            count = [sylco(word) for word in sentence.split()]
            total = np.sum(count)

            self.__totalSyllables[pos] = total

    def __generateFluencyParameters(self):
        # By intervention

        # speaking_time = stop_time - start_time
        diffStopStart = (self.__stop_time -
                         self.__start_time).reshape(-1, 1)

        path = './files/results/' + \
            str(self.audio) + '_P/rule3/' + str(self.audio) + \
            '_fluencyParameters.csv'

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(["speaker", "speakingTime",
                             "totalSyllables", "phonationTime"])

            start = self.__start_time[0]
            end = self.__stop_time[0]
            totalSyl = self.__totalSyllables[0]
            phonationTime = end - start

            previousSpeaker = self.__speaker[0]

            for pos, elem in enumerate(self.__speaker[1:], start=1):

                if (previousSpeaker != elem):
                    # Save the previous value

                    writer.writerow(
                        [previousSpeaker, end - start, totalSyl[0], phonationTime[0]])

                    start = self.__start_time[pos]
                    totalSyl = self.__totalSyllables[pos]
                    phonationTime = diffStopStart[pos]

                else:
                    totalSyl += self.__totalSyllables[pos]
                    phonationTime += diffStopStart[pos]

                end = self.__stop_time[pos]
                previousSpeaker = elem

    def __computeFormulas(self):
        path = './files/results/' + \
            str(self.audio) + '_P/rule3/' + str(self.audio) + \
            '_fluencyParameters.csv'

        fileParameters = readCSV(path, ",")

        speaker = fileParameters[:, 0].astype(np.int)
        speakingTime = fileParameters[:, 1].astype(np.float)
        totalSyllables = fileParameters[:, 2].astype(np.float)
        phonationTime = fileParameters[:, 3].astype(np.float)

        path = './files/results/' + \
            str(self.audio) + '_P/rule3/' + str(self.audio) + \
            '_fluencyMeasures.csv'

        with open(path, 'w', newline='') as file:

            writer = csv.writer(file)

            writer.writerow(["speaker", "averageSyllableDuration",
                             "articulationRate", "speechRate"])

            for x in np.arange(fileParameters.shape[0]):

                averageSyllableDuration = speakingTime[x] / totalSyllables[x]

                articulationRate = totalSyllables[x] / phonationTime[x]

                speechRate = totalSyllables[x] / speakingTime[x]

                writer.writerow(
                    [speaker[x], averageSyllableDuration, articulationRate, speechRate])

    def __detectDisfluenciesTranscript(self):

        self.__rowsTranscript = self.transcript.shape[0]

        #hesitationRepetitionWords = ("um", "eh", "eh,", "uh")

        #reservedWords = ("<cough>", "<laughter>", "<sigh>")

        pathDictionary = './files/results/' + \
            str(self.audio) + '_P/rule3/dictionaryAnswers.csv'

        with open(pathDictionary, 'w', newline='') as fileDictionary:
            self.__writer = csv.writer(fileDictionary)
            self.__writer.writerow(["start_time", "answer1", "answer2"])

            for pos, elem in enumerate(self.__values[:(self.__rowsTranscript-1)]):
                # Person mispeaks --> Not analysed

                # Repetitions and hesitations
                # Speech is cut-off
                # Unrecognizable words
                if (self.__speaker[pos] == participantCode):
                    self.__y = pos

                    if (self.__areHesitationWords()
                        or any(x in elem.split() for x in hesitationRepetitionWords)
                        or all(x not in elem for x in reservedWords) and "<" in elem and ">" in elem
                            or ("xxx" in elem)):

                        self.__getNextPyschologistAnswer()

    def __getNextPyschologistAnswer(self):

        for x in np.arange(self.__y, self.__rowsTranscript):
            if (self.__speaker[x] == therapistCode):

                if (x + 1 < self.__rowsTranscript and self.__speaker[x + 1] == therapistCode):
                    self.__writer.writerow(
                        [self.__start_time[x], self.__values[x].strip(), self.__values[x + 1].strip()])
                else:
                    self.__writer.writerow(
                        [self.__start_time[x], self.__values[x].strip(), ""])
                break

    def __areHesitationWords(self):

        s = self.__values[self.__y].split()

        for pos in np.arange(0, len(s) - 1):
            if (s[pos] == s[pos + 1]):
                return True

        return False
