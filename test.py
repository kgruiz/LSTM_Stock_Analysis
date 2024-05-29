import os
import re
import atexit
import pickle
import inspect
import warnings
import functools
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import matplotlib.pyplot as plt
import graphviz
from itertools import combinations
import time
import igraph as ig
import graphistry
from igraph import *
import igraph as ig
import pydot
import csv
from tabulate import tabulate as tb


do = 0


class model:

    def __init__(self):
        self.model = None
        self.modelInit()
        self.modelOps()

    def modelInit(self):

        import tensorflow as tf
        from tensorflow.python.platform import gfile
        import os
        import subprocess
        import tensorflow as tf
        from tensorflow.python.framework import graph_util
        from tensorflow.python.framework import graph_io
        import tensorflow as tf

        import tensorflow as tf
        import numpy as np
        from tensorflow import keras
        import subprocess
        from tensorflow import profiler

        from tensorflow import keras

        import tensorflow as tf

        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

        Sequential = keras.models.Sequential
        Dense = keras.layers.Dense
        LSTM = keras.layers.LSTM
        Dropout = keras.layers.Dropout
        Input = keras.layers.Input
        Model = keras.models.Model

        reqShape = (1, 60, 1)

        # Define the required shape
        reqShape = (1, 60, 1)

        # Create the NumPy array with zeros
        self.xArray = np.zeros(reqShape)

        # Fill the array with numbers ranging from 70 to 130
        self.xArray[:] = np.arange(70, 130).reshape(reqShape)

        yReqShape = (1, -1)
        self.y = [12]
        self.y = np.array(self.y)
        self.y = np.reshape(self.y, yReqShape)

        print(f"\nNow Training\n")

        self.model = Sequential()
        self.model.add(Input(shape=(self.xArray.shape[1], 1)))
        self.model.add(LSTM(128, return_sequences=True, activation="relu"))
        self.model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
        self.model.add(LSTM(64, return_sequences=False, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(25, activation="relu"))
        self.model.add(Dense(1))

        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def modelOps(self):

        self.model.fit(
            self.xArray,
            self.y,
            batch_size=1,
            epochs=1,
            verbose=1,
            # callbacks=[StackTraceCallback()],
        )  # Adjust epochs and batch size as needed

        self.model.save("./testModel.keras")

        # self.model = keras.models.load_model("./testModel.keras")

        # def convert_keras_model_to_dot(keras_model_path, output_dot):
        #     load_model = keras.models.load_model
        #     # Load the Keras model
        #     keras_model = load_model(keras_model_path)

        #     # Convert the Keras model to a TensorFlow graph
        #     tf_function = tf.function(lambda x: keras_model(x))
        #     concrete_func = tf_function.get_concrete_function(
        #         tf.TensorSpec(keras_model.inputs[0].shape, keras_model.inputs[0].dtype)
        #     )
        #     graph_def = concrete_func.graph.as_graph_def()

        #     # Export the computational graph to a DOT file
        #     with open(output_dot, "wt") as fh:
        #         print("digraph graphname {", file=fh)
        #         for node in graph_def.node:
        #             output_name = node.name
        #             print('  "' + output_name + '" [label="' + node.op + '"];', file=fh)
        #             for input_full_name in node.input:
        #                 parts = input_full_name.split(":")
        #                 input_name = parts[0]
        #                 print(
        #                     '  "' + input_name + '" -> "' + output_name + '";', file=fh
        #                 )
        #         print("}", file=fh)
        #         print(
        #             "Created DOT file '%s' for Keras model '%s'."
        #             % (output_dot, keras_model_path)
        #         )

        # keras_model_path = "./testModel.keras"
        # output_dot = "./output_model.dot"
        # convert_keras_model_to_dot(keras_model_path, output_dot)

        # def dot_to_png(dot_file, output_png):
        #     # Use the 'dot' command-line tool to convert the DOT file to a PNG image
        #     subprocess.run(["dot", "-Tpng", dot_file, "-o", output_png])

        # output_png = "./output_model.png"
        # dot_to_png(output_dot, output_png)

        raise SystemExit


class func:

    def __init__(self, name, filePath, className, line, time):

        self.name = name
        self.filePath = filePath
        self.line = int(line)
        self.className = className
        self.displayName = self.getDisplayName()
        self.time = time
        self.info = (
            self.name,
            self.filePath,
            self.line,
            self.className,
            self.displayName,
            self.time,
        )

    def getDisplayName(self):

        filePath = self.filePath

        # Split the string by '/'
        parts = filePath.split("/")

        # Get the last non-empty element
        prevFile = [part for part in parts if part][-1]

        # if self.name == "<module>":

        #     return "main"

        if self.name == "error_handler":

            return filePath + "/" + self.name + "~" + str(self.line)

        if self.className != "None":
            return "<" + self.className + "> " + self.name + "~" + str(self.line)

        else:
            return prevFile + "/" + self.name + "~" + str(self.line)

    def __eq__(self, other):
        if not isinstance(other, func):
            return False

        name = self.name == other.name
        filePath = self.filePath == other.filePath
        className = self.className == other.className
        line = self.line == other.line

        return (name) & (filePath) & (className) & (line)

    def __ne__(self, other):

        name = self.name == other.name
        filePath = self.filePath == other.filePath
        className = self.className == other.className
        line = self.line == other.line

        if (not name) | (not filePath) | (not className) | (not line):
            return True

        return False

    def __hash__(self):

        return hash((self.name, self.filePath, self.className, self.line))

    def __str__(self):

        return self.displayName


class visualizeModel:

    def __init__(self):
        self.makeGraph()

    def getFilesInDirectory(self, directoryPath):
        return [
            filename
            for filename in os.listdir(directoryPath)
            if os.path.isfile(os.path.join(directoryPath, filename))
        ]

    def getData(self):

        uniqueFuncts = set()

        tracesDict = dict()

        home = "/Users/kadengruizenga/CodingProjects/LSTM_Stock_Analysis/test.py"

        directoryPath = "./traces"
        filesInDirectory = self.getFilesInDirectory(directoryPath)

        for traceFile in tqdm(filesInDirectory, desc="Loading Files"):

            traceFileFull = directoryPath + "/" + traceFile

            with open(traceFileFull, "rb") as file:

                tracesFromFile = pickle.load(file)

                for time, stackTrace in tracesFromFile.items():

                    assert time not in tracesDict

                    tracesDict[time] = stackTrace

        for time, stackTrace in tqdm(
            tracesDict.items(), desc="Finding Unique Functions"
        ):

            origCaller = stackTrace[-3]
            filePath = re.search(r"File: (.*?),", origCaller).group(1)
            functionName = re.search(r"Function: (.*)", origCaller).group(1)

            if (filePath == home) & (functionName == "__init__"):

                continue  # ignorue files from compile and importing

            for funcCall in stackTrace:

                filePath = re.search(r"File: (.*?),", funcCall).group(1)
                className = re.search(r"Class: (.*?),", funcCall).group(1)
                line = re.search(r"Line: (.*?),", funcCall).group(1)
                functionName = re.search(r"Function: (.*)", funcCall).group(1)

                function = func(functionName, filePath, className, line, time)

                uniqueFuncts.add(function)

        for function in tqdm(uniqueFuncts, desc="Making displayNames Unique With File"):

            displayName = function.displayName

            # if name == "<module>":
            #     function.displayName = "main"
            #     continue

            nameCount = 0

            for funct in uniqueFuncts:

                if funct.displayName == displayName:
                    nameCount += 1

            if nameCount > 1:

                filePath = function.filePath

                # Split the string by '/'
                parts = filePath.split("/")

                # Get the last non-empty element
                prevFile = [part for part in parts if part][-1]

                function.displayName = prevFile + "/" + function.displayName

        for function in tqdm(uniqueFuncts, desc="Making displayNames Unique With Line"):

            displayName = function.displayName

            nameCount = 0

            for funct in uniqueFuncts:
                if "~" in funct.displayName:
                    mainDisName = re.search(r"(.*?)~", funct.displayName).group(1)
                else:
                    mainDisName = funct.displayName

                if mainDisName == displayName:
                    nameCount += 1

            if nameCount > 1 & ("~" not in displayName):

                line = function.line

                function.displayName = function.displayName + "~" + str(line)

        return uniqueFuncts, tracesDict

    def makeGraph(self):

        uniqueFuncts, tracesDict = self.getData()

        nodes = list(uniqueFuncts)

        edges = []

        for time, stackTrace in tqdm(tracesDict.items(), desc="Adding Edges"):

            for i, funcCall in enumerate(stackTrace):

                if funcCall != stackTrace[-1]:

                    filePath = re.search(r"File: (.*?),", funcCall).group(1)
                    className = re.search(r"Class: (.*?),", funcCall).group(1)
                    line = re.search(r"Line: (.*?),", funcCall).group(1)
                    functionName = re.search(r"Function: (.*)", funcCall).group(1)

                    function = func(functionName, filePath, className, line, time)

                    curNodeIdx = nodes.index(function)

                    curNode = nodes[curNodeIdx]

                    parentFuncCall = stackTrace[i + 1]

                    filePath = re.search(r"File: (.*?),", parentFuncCall).group(1)
                    className = re.search(r"Class: (.*?),", parentFuncCall).group(1)
                    line = re.search(r"Line: (.*?),", parentFuncCall).group(1)
                    functionName = re.search(r"Function: (.*)", parentFuncCall).group(1)

                    parentFunction = func(functionName, filePath, className, line, time)

                    parentNodeIdx = nodes.index(parentFunction)

                    parentNode = nodes[parentNodeIdx]

                    assert (curNode in nodes) & (parentNode in nodes)

                    edges.append((parentNode, curNode))

        with open("./edges", "wb") as file:
            pickle.dump(edges, file)

        with open("./nodes", "wb") as file:
            pickle.dump(nodes, file)
