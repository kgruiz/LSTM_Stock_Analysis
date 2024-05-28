import os
import re
import atexit
import pickle
import inspect
import warnings
import functools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import graphviz
from networkx.drawing.nx_agraph import to_agraph


class model:

    do = 0

    if do:

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

        # # Define a custom callback to capture stack traces
        # class StackTraceCallback(tf.keras.callbacks.Callback):
        #     import tensorflow as tf

        #     def __init__(self):
        #         import tensorflow as tf
        #         super(StackTraceCallback, self).__init__()

        #     def on_train_begin(self, logs=None):
        #         tf.profiler.experimental.start("logs")

        #     def on_train_end(self, logs=None):
        #         tf.profiler.experimental.stop()

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
        xArray = np.zeros(reqShape)

        # Fill the array with numbers ranging from 70 to 130
        xArray[:] = np.arange(70, 130).reshape(reqShape)

        yReqShape = (1, -1)
        y = [12]
        y = np.array(y)
        y = np.reshape(y, yReqShape)

        print(f"\nNow Training\n")

        model = Sequential()
        model.add(Input(shape=(xArray.shape[1], 1)))
        model.add(LSTM(128, return_sequences=True, activation="relu"))
        model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
        model.add(LSTM(64, return_sequences=False, activation="relu"))
        # model.add(Dropout(0.2))
        model.add(Dense(25, activation="relu"))
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mean_squared_error")

        model.fit(
            xArray,
            y,
            batch_size=1,
            epochs=1,
            verbose=1,
            # callbacks=[StackTraceCallback()],
        )  # Adjust epochs and batch size as needed

        model.save("./testModel.keras")

        # model = keras.models.load_model("./testModel.keras")

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

    def __init__(self, name, filePath, className, line):

        self.name = name
        self.filePath = filePath
        self.line = line
        self.className = className
        self.displayName = self.getDisplayName()

    def getDisplayName(self):

        filePath = self.filePath

        # Split the string by '/'
        parts = filePath.split("/")

        # Get the last non-empty element
        prevFile = [part for part in parts if part][-1]

        if self.name == "<module>":

            return "main"

        if self.name == "error_handler":

            return filePath + "/" + self.name + ":" + self.line

        if self.className != "None":
            return "<" + self.className + "> " + self.name + ":" + self.line

        else:
            return prevFile + "/" + self.name + ":" + self.line

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


class CAG:
    def __init__(self):
        self.graph = nx.DiGraph()

    def addNode(self, node):
        self.graph.add_node(node)

    def nodes(self):
        return list(self.graph.nodes())

    def delNode(self, node):
        self.graph.remove_node(node)

    def drawGraph(self):

        # Convert to AGraph object
        agraph = to_agraph(self.graph)

        # Add edge labels
        for u, v, data in self.graph.edges(data=True):
            agraph.add_edge(u, v, label=str(data.get("weight", "")))

        # Visualize the graph using Graphviz
        agraph.draw("graph.png", prog="dot")

        # Show the image with zoom
        img = plt.imread("graph.png")

        # Draw the original image
        plt.imshow(img)

        plt.axis("off")  # Hide axes
        plt.tight_layout()
        # plt.show()  # Display the graph


class visualizeModel:

    count = 0

    with open("./traces.pkl", "rb") as file:
        traces = pickle.load(file)

    with open("./fitData.pkl", "rb") as file:
        fitData = pickle.load(file)

    with open("./trainStepData.pkl", "rb") as file:
        trainStepData = pickle.load(file)

    tracesList = []
    tracesList.append(traces)
    fileP = "./traces"
    for i in range(2, 11):

        tempFile = fileP + str(i) + ".pkl"

        with open(tempFile, "rb") as file:
            tracesTemp = pickle.load(file)

        tracesList.append(tracesTemp)

    # with open("./tracesList", "wb") as file:
    #     pickle.dump(tracesList, file)

    uniqueFuncts = set()

    tracesDict = dict()

    for trace in tracesList:

        for funcTrace in trace.values():

            tracesDict[count] = funcTrace

            count += 1

    for traceD in tracesList:

        for key, value in traceD.items():

            for funcCall in value:

                filePath = re.search(r"File: (.*?),", funcCall).group(1)
                className = re.search(r"Class: (.*?),", funcCall).group(1)
                line = re.search(r"Line: (.*?),", funcCall).group(1)
                functionName = re.search(r"Function: (.*)", funcCall).group(1)

                function = func(functionName, filePath, className, line)

                uniqueFuncts.add(function)

    modelGraph = CAG()

    for function in uniqueFuncts:

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

    for function in uniqueFuncts:

        displayName = function.displayName

        nameCount = 0

        for funct in uniqueFuncts:
            if ":" in funct.displayName:
                mainDisName = re.search(r"(.*?):", funct.displayName).group(1)
            else:
                mainDisName = funct.displayName

            if mainDisName == displayName:
                nameCount += 1

        if nameCount > 1 & (":" not in displayName):

            line = function.line

            function.displayName = function.displayName + ":" + str(line)

    for function in uniqueFuncts:

        modelGraph.addNode(function)

    nodes = modelGraph.nodes()

    for node in nodes:

        print(f"Node display name: {node.displayName}, ID: {id(node)}")

    print(len(uniqueFuncts))

    a = 0

    for key, value in tracesDict.items():

        stack = []

        for funcCall in value:

            filePath = re.search(r"File: (.*?),", funcCall).group(1)
            className = re.search(r"Class: (.*?),", funcCall).group(1)
            line = re.search(r"Line: (.*?),", funcCall).group(1)
            functionName = re.search(r"Function: (.*)", funcCall).group(1)

            function = func(functionName, filePath, className, line)

            assert function in uniqueFuncts

            stack.append(function)

        nodes = modelGraph.nodes()

        for i, frame in enumerate(stack):

            curNodeIDx = nodes.index(frame)

            curNode = nodes[curNodeIDx]

            if ((curNode.name == "error_handler") | (curNode.name == "wrapper")) & (
                frame != stack[0]
            ):
                curNodeIDx = nodes.index(stack[i - 1])
                curNode = nodes[curNodeIDx]

            if frame != stack[-1]:

                nextNode = stack[i + 1]
                d = id(nextNode)
                te = 4
                print(a)
                parentNodeIDx = nodes.index(nextNode)
                parentNode = nodes[parentNodeIDx]

                if ((curNode.name == "error_handler") | (curNode.name == "wrapper")) & (
                    frame != stack[-2]
                ):
                    parentNodeIDx = nodes.index(stack[i + 2])
                    parentNode = nodes[parentNodeIDx]

                if modelGraph.graph.has_edge(parentNode, curNode):
                    modelGraph.graph.get_edge_data(parentNode, curNode)["weight"] += 1
                    a += 1

                else:
                    modelGraph.graph.add_edge(parentNode, curNode, weight=1)
                    a += 1

        assert (len(set(value)) <= len(set(nodes))) & (
            len(set(value)) <= len(uniqueFuncts)
        )

    for node in nodes:

        if (node.name == "error_handler") | (node.name == "wrapper"):

            modelGraph.delNode(node)

    modelGraph.drawGraph()
