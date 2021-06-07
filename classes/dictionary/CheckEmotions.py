import numpy as np
import csv

from util.codes import participant, negativeEmotionCode, positiveEmotionCode


class CheckEmotions:

    def __init__(self, audio, transcript):
        self.audio = audio
        self.transcript = transcript

    def checkPositiveEmotions(self):

        positiveEmotions = ["that’s good", "uh fun i", "that’s so good to hear",
                            "that sounds like a great situation",
                            "nice", "it’s pretty good", "awesome", "cool", "that’s it", "great",
                            "as uh charming", "emotional", "hilarious", "i like that", "it was good it was nice",
                            "really happy", "really happy i mean um", "that's so good to hear"]

        positiveEmotions = np.array(positiveEmotions)

        negativeEmotions = ["yeah i’m sorry", "i’m sorry", "yeah i'm sorry to hear that",
                            "i'm sorry to hear that", "that sucks", "yeah that sucks", "that sounds really hard"]

        negativeEmotions = np.array(negativeEmotions)

        # Rows
        self.__rowsTranscript = self.transcript.shape[0]

        self.__speaker = self.transcript[:, 2]
        self.__values = self.transcript[:, 3]

        self.__results = []

        for idx, elem in enumerate(self.__values):

            mask = [x == elem for x in positiveEmotions]

            if any(mask):
                # print(mask)
                emotion = positiveEmotions[mask]
                # Get next intervention of 'Participant'
                self.__getNextIntervention(
                    emotion[0], positiveEmotionCode, idx)
            else:
                mask = [x == elem for x in negativeEmotions]

                if any(mask):
                    emotion = negativeEmotions[mask]
                    # Get next intervention of 'Participant'
                    self.__getNextIntervention(
                        emotion[0], negativeEmotionCode, idx)

        path = './files/results/' + \
            str(self.audio) + '_P/checkEmotions.csv'

        finalResults = np.array(self.__results)

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(
                ["emotion", "type", "participantReaction"])
            writer.writerows(finalResults)

    def __getNextIntervention(self, emotion, type, idx):

        for pos in np.arange(idx, self.__rowsTranscript):

            if self.__speaker[pos] == participant:
                self.__results.append([emotion, type, self.__values[pos]])
