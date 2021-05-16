from util.readFile import readCSV, readTXT


def getCLNF_features(XXX):
    # Function to read XXX_CLNF_features.txt
    matrix = readTXT(getCorpusPath(XXX) + getCLNFPath(XXX) +
                     'features.txt', ',')
    return matrix


def getCLNF_features3D(XXX):
    # Function to read XXX_CLNF_features3D.txt
    matrix = readTXT(getCorpusPath(XXX) + getCLNFPath(XXX) +
                     'features3D.txt', ',')
    return matrix


def getCLNF_gaze(XXX):
    # Function to read XXX_CLNF_gaze.txt
    matrix = readTXT(getCorpusPath(XXX) + getCLNFPath(XXX) +
                     'gaze.txt', ',')
    return matrix


def getCLNF_hog(XXX):
    # Function to read XXX_CLNF_hog.bin (DOUBT: .bin or .txt)
    # matrix = readTXT(getCorpusPath(XXX) + getCLNFPath(XXX) +
    #                'hog.bin', ',')
    return matrix


def getCLNF_pose(XXX):
    # Function to read XXX_CLNF_pose.txt
    matrix = readTXT(getCorpusPath(XXX) + getCLNFPath(XXX) +
                     'pose.txt', ',')
    return matrix


def getCLNF_AUs(XXX):
    # Function to read XXX_CLNF_AUs.csv
    matrix = readCSV(getCorpusPath(XXX) + getCLNFPath(XXX) +
                     'AUs.csv', ',')
    return matrix


def readCOVAREP(XXX, delimiter):
    # Function to read XXX_COVAREP.csv
    matrix = readCSV(getCorpusPath(XXX) +
                     getInsidePath(XXX) + 'COVAREP.csv', delimiter)
    return matrix


def getFORMANT(XXX, delimiter):
    # Function to read XXX_FORMANT.csv
    matrix = readCSV(getCorpusPath(XXX) +
                     getInsidePath(XXX) + 'FORMANT.csv', delimiter)
    return matrix


def getTRANSCRIPT(XXX, delimiter):
    # Function to read XXX_TRANSCRIPT.csv
    matrix = readCSV(getCorpusPath(XXX) +
                     getInsidePath(XXX) + 'TRANSCRIPT.csv', delimiter)
    return matrix


def getAUDIOPath(XXX):
    return getCorpusPath(XXX) + getInsidePath(XXX) + 'AUDIO.wav'

##################### Example Function ############################


def readCSVFile():
    matrix = readCSV('../files/csv/dev_split_Depression_AVEC2017.csv', ',')
    print(matrix)
    print('Shape = ' + str(matrix.shape))


######################## Path Functions ###########################

def getCorpusPath(XXX):
    # Get the path of the XXX folder
    return 'files/corpus/' + str(XXX) + '_P/'


def getInsidePath(XXX):
    # Get inside path of the XXX folder
    return str(XXX) + '_'


def getCLNFPath(XXX):
    # Get the initial name of the CLNF files
    return str(XXX) + '_' + 'CLNF_'

################################################################
