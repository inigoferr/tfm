import time

import numpy as np
from opensmile.core.define import FeatureSet
import pandas as pd

import audiofile
import opensmile

from manageFiles.getFile import getAUDIOPath, getTRANSCRIPT

"""
    | Feature + Value |

    [] F1: Lowering of pitch in speech signal --> Back-channel
    [] F2: Raised loudness in speech signal --> Back-channel
    [] F3: Disfluency in speech signal --> Question to the user
    [] F4: A region of pitch less than the 26th-percentile pitch level and continuing for at least 110 milliseconds --> Back-channel
    [X] F5: After at least 700 milliseconds of speech --> Back-channel
    [X] F6: After 700 milliseconds wait --> Back-channel
    [] F7: Not output back-channel feedback within the preceding 800 milliseconds --> Back-channel
"""

# Participants
therapist = 'Ellie'
therapistCode = 0
user = 'Participant'
userCode = 1

# Number of Corpus
startCorpus = 300
endCorpus = 306

# Variables XXX_TRANSCRIPT.csv
start_time = 0
stop_time = 1
speaker = 2
value = 3

# Columns for the matrix 'speech'
posDuration = 0
posSpeaker = 1
posNumLines = 2
posFirstLine = 3


def startAnalysis():

    # Analyse the TRANSCRIPT
    conversations = np.arange(startCorpus, startCorpus + 1)

    for corpus in conversations:
        matrix = getTRANSCRIPT(corpus, '\t')  # \t = 'tab delimiter'

        # Analyse Pitch region less than 26th percentile pitch level...
        analyseF4(corpus, matrix)

        # After at least 700 millseconds of speech
        #analyseAfter700MillisecondsOfSpeech(corpus, matrix)

        # After 700 milliseconds of wait (= silence)
        #analyseAfter700MillisecondsOfWait(corpus, matrix)


######################## F4 #########################################

def analyseF4(corpus, matrix):
    print("----------------------------------------------------------------------")
    print("Region of pitch less than the 26th-percentile pitch level and continuing for at least 110 milliseconds " + str(corpus) + "_P")
    print("----------------------------------------------------------------------")

    file = getAUDIOPath(corpus)

    #signal, sampling_rate = audiofile.read('test.wav', always_2d=True)
    signal, sampling_rate = audiofile.read(file, always_2d=True)

    print("Sampling_rate " + str(sampling_rate))

    smile = opensmile.Smile(
        feature_set='config/config.conf',
        loglevel=2,
        logfile='smile.log',
    )

    print("Feature Names = " + str(smile.feature_names))

    df = smile.process_signal(
        signal,
        sampling_rate
    )

    print("Finalizado Signal...")

    df.to_csv('analysisPercentilesDuration.csv', sep='\t')

    print("CSV made...")

    with open('./smile.log', 'r') as fp:
        log = fp.readlines()

    print("THE END!!")
    # print(log)


######################## F5 #########################################

def analyseAfter700MillisecondsOfSpeech(corpus, matrix):
    print("----------------------------------------------------------------------")
    print("Analysis After 700 milliseconds of Speech " + str(corpus) + "_P")
    print("----------------------------------------------------------------------")

    rows, columns = matrix.shape

    # Compute the duration of speech

    # Structure of one row of the matrix 'speech' ['durationOfTheSpeech' 'codeOfTheSpeaker' 'numLinesInTRANSCRIPT' 'firstLineInTRANSCRIPT' ]
    speech = np.zeros((rows, 4))

    accumulatedDuration = 0
    previousSpeaker = matrix[0, speaker]
    numLines = 0
    firstLine = 2

    for x in np.arange(0, rows):
        actualSpeaker = matrix[x, speaker]

        # Only interested on user's speech, not therapist's
        if (actualSpeaker == user):
            # The speaker talking is different from the previous one
            if (actualSpeaker != previousSpeaker):
                # Save the speech of the previousSpeaker
                saveSpeechInRow(x-1, speech, accumulatedDuration,
                                previousSpeaker, numLines, firstLine)

                # Reset values
                accumulatedDuration = (float(matrix[x, stop_time]) -
                                       float(matrix[x, start_time]))
                numLines = 1
                firstLine = x + 2
            else:  # The speaker talking is the same from the previous sentence
                accumulatedDuration += (float(matrix[x, stop_time]) -
                                        float(matrix[x, start_time]))
                numLines += 1

        previousSpeaker = actualSpeaker

    # Save the last sentence if necessary
    saveSpeechInRow(rows - 1, speech, accumulatedDuration,
                    previousSpeaker, numLines, firstLine)

    # Get rows with duration more than 700 milliseconds
    maskNoZeros = (speech[:, posDuration] > 7.0)

    print(speech[maskNoZeros])

    print("----------------------------------------------------------------------")
    print("\n")


def saveSpeechInRow(row, speech, accumulatedDuration, previousSpeaker, numLines, firstLine):
    speech[row, posDuration] = accumulatedDuration
    speech[row, posSpeaker] = therapistCode if (
        previousSpeaker == therapist) else userCode
    speech[row, posNumLines] = numLines
    speech[row, posFirstLine] = firstLine

################################ F6 ###############################


def analyseAfter700MillisecondsOfWait(corpus, matrix):
    print("----------------------------------------------------------------------")
    print("Analysis After 700 milliseconds Of Wait for " + str(corpus) + "_P")
    print("----------------------------------------------------------------------")
    rows, columns = matrix.shape
    # Compute the silences,
    # if silence[z] == 0 --> Means that the speaker of 'z-1' sentence and the 'z' sentence was the same
    # if silence[z] < 0 --> Means that the speakers spoke at the same time or there is an error in the Transcript

    silences = np.zeros(rows - 1)
    before = matrix[0, :]
    count = 0
    for x in np.arange(1, rows):
        now = matrix[x, :]

        if (now[speaker] != before[speaker]):
            silences[count] = float(
                now[start_time]) - float(before[stop_time])

        count += 1
        before = now

    # Average Silences
    # averageSilence = np.average(silences)
    # print("Average Silence = " + str(averageSilence))

    # Rows where silence was 700 milliseconds or more
    mask700_More = silences >= 7.0
    positions700_More = np.array(np.where(mask700_More)) + 1
    rows700OrMore = silences[mask700_More]

    print("Pair of sentences where the silence was 700 milliseconds or more")
    print("######################################################################")
    showPairSentencesSilence(
        matrix, positions700_More, rows700OrMore, corpus)
    print("######################################################################")
    print("\n")


def showPairSentencesSilence(matrix, positions, seconds, corpus):
    i = 0
    for z in positions:
        for x in z:
            print(matrix[x - 1, speaker] + ': ' + matrix[x - 1, value])
            print(matrix[x, speaker] + ': ' + matrix[x, value])
            print("Seconds: " + str(np.around(seconds[i], 2)) + " seconds")
            print("Lines of " + str(corpus) +
                  "_TRANSCRIPT.csv: " + str(x+1) + " & " + str(x+2))
            print("-----------------------------------------------------")
            i += 1

######################################################################################
