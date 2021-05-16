import csv
import numpy as np
import matlab.engine


def readCSV(file, delimiter):
    with open(file) as csv_file:
        # Create the CSV_Reader
        csv_reader = csv.reader(csv_file, delimiter=delimiter)

        matrix = []
        # Read the 1ºLine
        next(csv_reader)
        # Read the rest of the rows and add to the matrix
        for row in csv_reader:
            matrix.append(row)

        return np.array(matrix)


def readTXT(file, delimiter):
    with open(file, "r") as reader:

        matrix = []
        # Read the 1ºLine
        reader.readline()
        for row in reader:
            currentline = row.split(delimiter)
            matrix.append(currentline)

        return np.array(matrix)


def readHOG(users, hog_data_dir):
    # Read HOG in a binary file
    eng = matlab.engine.start_matlab()
    # the function return [hog_data, valid_inds, vid_id]
    return eng.Read_HOG_files(users, hog_data_dir)
