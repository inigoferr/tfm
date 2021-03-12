import opensmile
from manageFiles.getFile import getAUDIOPath


def tryOpenSmile():
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    audioPath = getAUDIOPath(300)
    print(audioPath)
    y = smile.process_file(audioPath)
    print(y)
