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

        # *****************************************************************
        # Start times of Transcript
        # Participant
        [result.append([s, lParticipant])
         for s in start_time[speaker == participant]]

        # *****************************************************************
        # Stop times of Transcript
        [result.append([s, lSpeaking]) for s in stop_time]

        # *****************************************************************
        # Therapist
        # Therapist lines
        maskTherapist = (speaker == therapist)
        startTimeTherapist = start_time[maskTherapist]
        valueTherapist = value[maskTherapist]

        # First rows of the therapist that are always an introduction
        t = 0
        for pos, sp in enumerate(speaker):
            if (sp == participant):
                t = pos
                break

        [result.append([s, lFirstRowsTherapist])
         for s in startTimeTherapist[:t]]

        # Rest of start_time of the therapist
        for pos in np.arange(t, startTimeTherapist.shape[0]):
            if (valueTherapist[pos].strip().startswith(wordsQuestions)):
                result.append([startTimeTherapist[pos], lQuestion])
            else:
                result.append([startTimeTherapist[pos], lBackChannel])
        # *****************************************************************

        # FrameTimes
        stp = self.totalFrames * self.frameSize
        frameTime = np.arange(
            start=0.0, stop=stp, step=self.frameSize)

        # *****************************************************************
        # Frame times before start_time[0]
        maskStartTimeZero = frameTime < start_time[0]
        [result.append([np.round(f, 3), lSilence])
         for f in frameTime[maskStartTimeZero]]

        # *****************************************************************
        # Frame time after start_time[0]

        rows = start_time.shape[0]
        for pos in np.arange(rows - 1):
            maskStart = start_time[pos] < frameTime
            maskStop = frameTime < stop_time[pos]

            # lSpeaking
            [result.append([np.round(f, 3), lSpeaking])
             for f in frameTime[maskStart & maskStop]]

            # lSilence
            maskActualStop = stop_time[pos] < frameTime
            maskNextStart = frameTime < start_time[pos + 1]

            [result.append([np.round(f, 3), lSilence])
             for f in frameTime[maskActualStop & maskNextStart]]

        # Last row start_time[-1] and stop_time[-1]
        maskStart = start_time[-1] < frameTime
        maskStop = frameTime < stop_time[-1]
        [result.append([np.round(f, 3), lSpeaking])
            for f in frameTime[maskStart & maskStop]]
        # *****************************************************************

        # *****************************************************************
        # Frame times after stop_time[-1]
        maskLastStopTime = stop_time[-1] < frameTime
        [result.append([np.round(f, 3), lSilence])
         for f in frameTime[maskLastStopTime]]
        # *****************************************************************

        # Transform to array
        result = np.unique(np.array(result), axis = 0) 
        # Remove uniques because the transcript can have a stop time and start time error

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
