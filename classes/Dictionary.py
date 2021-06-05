import numpy as np
import csv

from util.readFile import readCSV
from util.codes import therapist, therapistCode, participant


class Dictionary:

    def __init__(self, audio):
        self.audio = audio

    def generateDictionary(self):

        # Initialise dictionary
        self.__dictionary = []

        # Obtain answers from Rule 1 and Rule 2
        self.__getAnswersRule1And2()

        # Obtain answers from Rule 3
        self.__getAnswersRule3()

        # Obtain answers from Rule 4
        self.__getAnswersRule4()

        # Obtain answers from Rule 5
        self.__getAnswersRule5()

        # Obtain answers from Rule 6
        self.__getAnswersRule6()

        # Obtain answers from Rule 7 --> No Results

        # Save answers in XXX_Dictionary.csv
        self.__saveDictionary()

    def __getAnswersRule1And2(self):
        path1 = './files/results/' + \
            str(self.audio) + '_P/rule1/dictionaryAnswers.csv'

        file1 = readCSV(path1, ",")

        path2 = './files/results/' + \
            str(self.audio) + '_P/rule2/dictionaryAnswers.csv'

        file2 = readCSV(path2, ",")

        files = np.array([file1, file2])

        for file in files:
            if (file.shape[0] > 0):
                start_time = file[:, 2].reshape(-1, 1)
                speaker = file[:, 3].reshape(-1, 1).astype(np.int)
                answers = file[:, 4].reshape(-1, 1)

                results = np.hstack((start_time, speaker, answers))
                fltr = np.asarray([str(therapistCode)])

                newResults = results[np.in1d(results[:, 1], fltr)]
                newResults = np.unique(newResults, axis=0)[:, [0, 2]]

                [self.__dictionary.append(x) for x in newResults]

    def __getAnswersRule3(self):
        path = './files/results/' + \
            str(self.audio) + '_P/rule3/dictionaryAnswers.csv'

        file = readCSV(path, ",")

        if (file.shape[0] > 0):

            f1 = file[file[:, 2] == ""]
            st1 = f1[:, 0].reshape(-1, 1)
            a1 = f1[:, 1].reshape(-1, 1)
            r1 = np.hstack((st1, a1))

            f2 = file[file[:, 2] != ""]
            st2 = f2[:, 0].reshape(-1, 1)
            a2 = f2[:, 2].reshape(-1, 1)
            r2 = np.hstack((st2, a2))

            results = np.vstack((r1, r2))
            results = np.unique(results, axis=0)

            [self.__dictionary.append(x) for x in results]

    def __getAnswersRule4(self):

        path = './files/results/' + \
            str(self.audio) + '_P/rule4/' + str(self.audio) + \
            '_region26Pitch.csv'

        file = readCSV(path, ",")

        if (file.shape[0] > 0):

            start_time = file[:, 1].reshape(-1, 1)
            answers = file[:, 2].reshape(-1, 1)

            results = np.hstack((start_time, answers))
            results = np.unique(results, axis=0)

            [self.__dictionary.append(x) for x in results]

    def __getAnswersRule5(self):
        path = './files/results/' + \
            str(self.audio) + '_P/rule5/dictionaryAnswersPyschologist.csv'

        file = readCSV(path, ",")

        if (file.shape[0] > 0):

            start_time = file[:, 2].reshape(-1, 1)
            answers = file[:, 3].reshape(-1, 1)

            results = np.hstack((start_time, answers))
            results = np.unique(results, axis=0)

            [self.__dictionary.append(x) for x in results]

    def __getAnswersRule6(self):

        path = './files/results/' + \
            str(self.audio) + '_P/rule6/dictionarySentences.csv'

        file = readCSV(path, ",")

        speaker1 = file[:, 2].reshape(-1, 1)

        start_time = file[:, 4].reshape(-1, 1)

        speaker2 = file[:, 6].reshape(-1, 1)

        answer2 = file[:, 7].reshape(-1, 1)

        matrix = np.hstack((speaker1, start_time, speaker2, answer2))

        for elem in matrix:
            if (elem[0] == participant and elem[2] == therapist):
                self.__dictionary.append((elem[1], elem[3]))

    def __saveDictionary(self):

        path = './files/results/' + \
            str(self.audio) + '_P/' + \
            str(self.audio) + '_Dictionary.csv'

        uniqueAnswers = np.unique(np.array(self.__dictionary), axis=0)

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(["start_time", "answer"])

            writer.writerows(uniqueAnswers)
