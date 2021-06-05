import numpy as np

from classes.evaluateRules.EvalRule import EvalRule
from util.codes import hesitationRepetitionWords, reservedWords


class EvalRule3(EvalRule):

    def evaluateRule(self):
        if False:  # self.__isDisfluency()  # Question to the user
            # We should ask a lQuestion
            self.__writer.writerow(
                [np.round(frame*self.__frameSize, 3), lQuestion])
            return True
        else:
            return False

    def __isDisfluency(self, elem):

        return (self.__areHesitationWords()
                or any(x in elem.split() for x in hesitationRepetitionWords)
                or all(x not in elem for x in reservedWords) and "<" in elem and ">" in elem
                or ("xxx" in elem))

    def __areHesitationWords(self, x):

        s = self.__values[x].split()

        for pos in np.arange(0, len(s) - 1):
            if (s[pos] == s[pos + 1]):
                return True

        return False
