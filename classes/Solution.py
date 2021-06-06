import numpy as np
import csv

from util.codes import therapist, participant, lFirstRowsTherapist, lBackChannel, lQuestion, lParticipant, lSilence, lSpeaking, wordsQuestions


class Solution:

    def __init__(self, audio, transcript, totalFrames, frameSize):
        self.audio = audio
        self.transcript = transcript
        self.totalFrames = totalFrames
        self.frameSize = frameSize

    def generateSolution(self):

        start_time = self.transcript[:, 0].astype(np.float)
        stop_time = self.transcript[:, 1].astype(np.float)
        speaker = np.array([item.strip()
                            for item in self.transcript[:, 2]])
        value = self.transcript[:, 3]

        result = []

        # Start times of Transcript
        # Participant
        [result.append([s, lParticipant])
         for s in start_time[speaker == participant]]

        # Therapist
        # Therapist lines
        maskTherapist = (speaker == therapist)
        sTherapist = start_time[maskTherapist]
        vTherapist = value[maskTherapist]

        # First rows of the therapist that are always an introduction
        t = 0
        for pos, sp in enumerate(speaker):
            if (sp == participant):
                t = pos
                break
        [result.append([s, lFirstRowsTherapist]) for s in sTherapist[:t]]

        # Rest of rows
        for pos in np.arange(t, sTherapist.shape[0]):
            if (vTherapist[pos].strip().startswith(wordsQuestions)):
                result.append([sTherapist[pos], lQuestion])
            else:
                result.append([sTherapist[pos], lBackChannel])

        # FrameTimes
        stp = self.totalFrames * self.frameSize
        frameTime = np.arange(
            start=0.0, stop=stp, step=self.frameSize)

        # Frame times before start_time[0]
        maskStartTimeZero = frameTime < start_time[0]
        [result.append([np.round(f, 3), lSilence])
         for f in frameTime[maskStartTimeZero]]

        rows = start_time.shape[0]
        for pos in np.arange(rows):
            maskStart = start_time[pos] < frameTime
            maskStop = frameTime < stop_time[pos]

            [result.append([np.round(f, 3), lSpeaking])
             for f in frameTime[maskStart & maskStop]]

        # Frame times after stop_time[-1]
        maskLastStopTime = stop_time[-1] < frameTime
        [result.append([np.round(f, 3), lSilence])
         for f in frameTime[maskLastStopTime]]

        # Transform to array
        result = np.array(result)

        # Save Solution
        self.__saveSolution(result)

    def __saveSolution(self, result):

        # Save the solution in the solution.csv
        solution = result[np.argsort(result[:, 0])]
        path = './files/results/' + \
            str(self.audio) + '_P/solution.csv'

        with open(path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["frameTime", "label"])
            writer.writerows(solution)
