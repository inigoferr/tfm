
from classes.Evaluation import Evaluation
from classes.Dictionary import Dictionary
import numpy as np

from classes.rules.Rule7 import Rule7
from classes.rules.Rule4 import Rule4
from classes.rules.Rule3 import Rule3
from classes.rules.Rule2 import Rule2
from classes.rules.Rule1 import Rule1
from classes.rules.Rule5 import Rule5
from classes.rules.Rule6 import Rule6
from util.getFile import getTRANSCRIPT

"""
    | Feature + Value |

    [X] Rule 1: Lowering of pitch in speech signal --> Back-channel
    [X] Rule 2: Raised loudness in speech signal --> Back-channel
    [X] Rule 3: Disfluency in speech signal --> Question to the user
    [X] Rule 4: A region of pitch less than the 26th-percentile pitch level and continuing for at least 110 milliseconds --> Back-channel
    [X] Rule 5: After at least 700 milliseconds of speech --> Back-channel
    [X] Rule 6: After 700 milliseconds wait --> Back-channel
    [X] Rule 7: Not output back-channel feedback within the preceding 800 milliseconds --> Back-channel
"""

# Number of Corpus
startCorpus = 300
endCorpus = 332


def startAnalysis():

    # Analyse the TRANSCRIPT
    #conversations = [300]
    conversations = np.arange(startCorpus, endCorpus + 1)

    for audio in conversations:
        if (audio != 316):  # There is no data for 316_P
            print("Audio = " + str(audio))

            transcript = getTRANSCRIPT(audio, '\t')  # \t = 'tab delimiter'

            """
            # Analyse Rule 1
            r1 = Rule1(audio, transcript)
            r1.analyseRule()

            # Analyse Rule 2
            r2 = Rule2(audio, transcript)
            r2.analyseRule()

            # Analyse Rule 4
            r4 = Rule4(audio, transcript)
            r4.analyseRule()

            # After at least 700 millseconds of speech
            r5 = Rule5(audio, transcript)
            r5.analyseRule()

            # After 700 milliseconds of wait (= silence)
            r6 = Rule6(audio, transcript)
            r6.analyseRule()

            # Analyse Rule 7
            r7 = Rule7(audio, transcript)
            r7.analyseRule()

            # Analyse Rule 3
            r3 = Rule3(audio, transcript)
            r3.analyseRule()
"""
            dictionary = Dictionary(audio)
            dictionary.generateDictionary()


"""
            evaluation = Evaluation(audio, transcript)
            evaluation.evaluate()

"""
