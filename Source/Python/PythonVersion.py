import os
import shutil
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import yfinance as yf
from typing import Match
from tensorflow import keras
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM
Dropout = keras.layers.Dropout
Input = keras.layers.Input


sns.set_style("darkgrid")  # adds seaborn style to charts, eg. grid
plt.style.use("dark_background")  # inverts colors to dark theme

warnings.filterwarnings("ignore")


global show
global techList
global stockData

show = 0  # 0 = don't show and save, 1 = show don't and save, 2 =  show and save, 3 = neither

# The tech stocks we'll use for this analysis
techList = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA", "NVDA"]


def delPast():

    savePath = "./Output/"

    # Check if the directory exists before attempting to delete it
    if os.path.exists(savePath):
        # Use shutil.rmtree to remove the directory and its contents recursively
        shutil.rmtree(savePath)


delPast()


def showSave(fig, modelOutput=False):

    # Check the type of plot
    if isinstance(fig, sns.axisgrid.JointGrid):
        # It's a seaborn jointplot
        mainTitle = fig.figure.suptitle.get_text()
    elif isinstance(fig, sns.axisgrid.PairGrid):
        # It's a seaborn pairplot
        # PairGrid doesn't have a single title, so you might need to access individual subplot titles
        mainTitle = "pairGrid"
    elif isinstance(fig, sns.matrix.ClusterGrid):
        # It's a seaborn clustermap
        mainTitle = fig.figure.suptitle.get_text()
    else:
        # It's likely a matplotlib plot
        # Get the title from the main supertitle attribute
        mainTitle = fig._suptitle._text

    if modelOutput:
        fileRoot = "./Output/ModelOutput/"
        fileName = fileRoot + mainTitle.replace(" ", "")
    else:
        fileRoot = "./Output/AnalysisOutput/"
        fileName = fileRoot + mainTitle.replace(" ", "")

    global show

    match show:

        case 0:

            if not os.path.exists(fileRoot):
                os.makedirs(fileRoot)

            plt.savefig(fileName)

        case 1:

            plt.show()

        case 2:

            if not os.path.exists(fileRoot):
                os.makedirs(fileRoot)

            plt.savefig(fileName)
            plt.show()

        case _:

            pass

    fig.clf()


def visualiseDict(d, lvl=0):

    # go through the dictionary alphabetically
    for k in sorted(d):

        # print the table header if we're at the beginning
        if lvl == 0 and k == sorted(d)[0]:
            print("{:<25} {:<15} {:<10}".format("KEY", "LEVEL", "TYPE"))
            print("-" * 79)

        indent = "  " * lvl  # indent the table to visualise hierarchy
        t = str(type(d[k]))

        # print details of each entry
        print("{:<25} {:<15} {:<10}".format(indent + str(k), lvl, t))

        # if the entry is a dictionary
        if isinstance(d[k], dict):
            # visualise THAT dictionary with +1 indent
            visualiseDict(d[k], lvl + 1)


# class analyze1:

#     global show
#     global techList
#     global stockData

#     # Set up End and Start times for data grab
#     end = datetime.now()
#     start = datetime(end.year - 1, end.month, end.day)

#     # Dictionary to store dataframes for each stock
#     stockData = {}

#     # Download data for each stock and store in the dictionary
#     for stock in techList:
#         stockData[stock] = yf.download(stock, start, end, progress=False)

#     # Names of the corresponding companies
#     companyNames = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON", "TESLA"]

#     # Add a column 'company_name' to each dataframe to identify the company
#     for companyDF, companyName in zip(stockData.values(), companyNames):
#         companyDF["companyName"] = companyName

#     # List of downloaded company dataframes
#     companyList = [stockData[stock] for stock in techList]
#     # Concatenate all dataframes into a single dataframe
#     df = pd.concat(companyList, axis=0)

#     del companyList

#     # Adj Close accounts for histoprical dividends, splits, ec.

#     # Let's see a historical view of the closing price
#     fig = plt.figure(figsize=(15, 10))
#     fig.suptitle("Adj Closing Prices")
#     plt.subplots_adjust(top=1.25, bottom=1.2)

#     for i, (company, companyData) in enumerate(stockData.items()):
#         plt.subplot(3, 2, i + 1)
#         companyData["Adj Close"].plot()
#         plt.ylabel("Adj Close")
#         plt.xlabel(None)
#         plt.title(f"Closing Price of {company}")

#     fig.tight_layout()

#     showSave(fig)

#     # Now let's plot the total volume of stock being traded each day
#     fig = plt.figure(figsize=(15, 10))
#     fig.suptitle("Sales Volumes")
#     plt.subplots_adjust(top=1.25, bottom=1.2)

#     for i, (company, companyData) in enumerate(stockData.items()):
#         plt.subplot(3, 2, i + 1)
#         companyData["Volume"].plot()
#         plt.ylabel("Volume")
#         plt.xlabel(None)
#         plt.title(f"Sales Volume for {company}")

#     fig.tight_layout()

#     showSave(fig)

#     maDay = [10, 20, 50]

#     # Calculate moving averages and RSI for each company
#     for ma in maDay:
#         for companyName, companyData in stockData.items():
#             columnNameMa = f"MA for {ma} days"
#             companyData[columnNameMa] = companyData["Adj Close"].rolling(ma).mean()

#     # Calculate RSI for each company
#     rsiPeriod = 14  # RSI period
#     for companyName, companyData in stockData.items():
#         delta = companyData["Adj Close"].diff(1)
#         gain = delta.where(delta > 0, 0)
#         loss = -delta.where(delta < 0, 0)

#         avgGain = gain.rolling(window=rsiPeriod, min_periods=1).mean()
#         avgLoss = loss.rolling(window=rsiPeriod, min_periods=1).mean()

#         rs = avgGain / avgLoss
#         rsi = 100 - (100 / (1 + rs))

#         companyData["RSI"] = rsi

#     fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(30, 20))
#     fig.suptitle("Stock Analysis")

#     # Function to plot the data for a given company
#     def plotCompanyData(ax, stockSymbol, companyData):
#         companyData[
#             ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days", "RSI"]
#         ].plot(ax=ax)
#         ax.set_title(f"Closing Price and Indicators for {stockSymbol}")
#         ax.legend()

#     # Plot data for each company on separate graphs
#     for ax, (stockSymbol, companyData) in zip(axes.flatten(), stockData.items()):
#         plotCompanyData(ax, stockSymbol, companyData)

#     fig.tight_layout(rect=[0, 0.03, 1, 0.95])
#     showSave(fig)

#     ma_day = [10, 20, 50]

#     for ma in ma_day:
#         for companyName, companyData in stockData.items():
#             columnName = f"MA for {ma} days"
#             companyData[columnName] = companyData["Adj Close"].rolling(ma).mean()

#     fig, axes = plt.subplots(nrows=3, ncols=2)
#     fig.suptitle("Adj Closing Prices w MA")
#     fig.set_figheight(10)
#     fig.set_figwidth(15)

#     stockData["AAPL"][
#         ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
#     ].plot(ax=axes[0, 0])
#     axes[0, 0].set_title("APPLE")

#     stockData["GOOG"][
#         ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
#     ].plot(ax=axes[0, 1])
#     axes[0, 1].set_title("GOOGLE")

#     stockData["MSFT"][
#         ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
#     ].plot(ax=axes[1, 0])
#     axes[1, 0].set_title("MICROSOFT")

#     stockData["AMZN"][
#         ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
#     ].plot(ax=axes[1, 1])
#     axes[1, 1].set_title("AMAZON")

#     stockData["TSLA"][
#         ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
#     ].plot(
#         ax=axes[2, 0]
#     )
#     axes[2, 0].set_title("TESLA")
    
#     stockData["NVDA"][
#         ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
#     ].plot(
#         ax=axes[2, 1]
#     )
#     axes[2, 1].set_title("NVDA")

#     fig.tight_layout()

#     showSave(fig)

#     # We'll use pct_change to find the percent change for each day
#     for companyName, companyData in stockData.items():
#         companyData["Daily Return"] = companyData["Adj Close"].pct_change(
#             fill_method=None
#         )

#     # Then we'll plot the daily return percentage
#     fig, axes = plt.subplots(nrows=3, ncols=2)
#     fig.suptitle("Daily Returns")
#     fig.set_figheight(10)
#     fig.set_figwidth(15)

#     stockData["AAPL"]["Daily Return"].plot(
#         ax=axes[0, 0], legend=True, linestyle="--", marker="o"
#     )
#     axes[0, 0].set_title("APPLE")

#     stockData["GOOG"]["Daily Return"].plot(
#         ax=axes[0, 1], legend=True, linestyle="--", marker="o"
#     )
#     axes[0, 1].set_title("GOOGLE")

#     stockData["MSFT"]["Daily Return"].plot(
#         ax=axes[1, 0], legend=True, linestyle="--", marker="o"
#     )
#     axes[1, 0].set_title("MICROSOFT")

#     stockData["AMZN"]["Daily Return"].plot(
#         ax=axes[1, 1], legend=True, linestyle="--", marker="o"
#     )
#     axes[1, 1].set_title("AMAZON")

#     stockData["TSLA"]["Daily Return"].plot(
#         ax=axes[2, 0], legend=True, linestyle="--", marker="o"
#     )
#     axes[2, 1].set_title("NVDA")
#     stockData["NVDA"]["Daily Return"].plot(
#         ax=axes[2, 1], legend=True, linestyle="--", marker="o"
#     )
#     axes[2, 1].set_title("NVDA")

#     fig.tight_layout()

#     showSave(fig)

#     fig = plt.figure(figsize=(12, 9))
#     fig.suptitle("Daily Returns Histogram")

#     for i, (companyName, companyData) in enumerate(stockData.items()):
#         plt.subplot(3, 2, i + 1)
#         companyData["Daily Return"].hist(bins=50)
#         plt.xlabel("Daily Return")
#         plt.ylabel("Counts")
#         plt.title(f"{companyName}")

#     fig.tight_layout()

#     showSave(fig)


# class analyze2:

#     # ***independent program break 1/2

#     global show
#     global techList
#     global stockData

#     # Set up End and Start times for data grab
#     end = datetime.now()
#     start = datetime(end.year - 1, end.month, end.day)

#     # Fetch historical data for tech stocks
#     closingDf = yf.download(techList, start=start, end=end, progress=False)["Adj Close"]

#     # Make a new tech returns DataFrame
#     techRets = closingDf.pct_change()
#     techRets.head()

#     fig = sns.jointplot(
#         x="GOOG", y="GOOG", data=techRets, kind="scatter", color="seagreen"
#     )

#     fig.figure.suptitle("GOOG vs GOOG Joint")

#     plt.tight_layout()

#     showSave(fig.figure)
#     fig = sns.jointplot(x="GOOG", y="MSFT", data=techRets, kind="scatter")

#     fig.figure.suptitle("GOOG vs MSFT Joint")

#     plt.tight_layout()

#     showSave(fig.figure)

#     fig = sns.pairplot(techRets, kind="reg")

#     fig.figure.suptitle("techPairPlot")

#     showSave(fig.figure)

#     fig = plt.figure(figsize=(12, 5))  # Adjusted height to make it shorter
#     fig.suptitle("Heatmap")

#     ax1 = plt.subplot(1, 2, 1, aspect="equal")  # Set aspect ratio to equal
#     sns.heatmap(techRets.corr(), annot=True, cmap="summer")
#     plt.title("Correlation of return")

#     ax2 = plt.subplot(1, 2, 2, aspect="equal")  # Set aspect ratio to equal
#     sns.heatmap(closingDf.corr(), annot=True, cmap="summer")
#     plt.title("Correlation of closing price")

#     plt.tight_layout()  # Adjust layout for better spacing

#     showSave(fig)

#     rets = techRets.dropna()

#     area = np.pi * 20  # size of scatter plot dots

#     fig = plt.figure(figsize=(10, 8))
#     fig.suptitle("Risk vs Return")
#     plt.scatter(rets.mean(), rets.std(), s=area)
#     plt.xlabel("Expected return")
#     plt.ylabel("Risk")

#     for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
#         plt.annotate(
#             label,
#             xy=(x, y),
#             xytext=(50, 50),
#             textcoords="offset points",
#             ha="right",
#             va="bottom",
#             arrowprops=dict(
#                 arrowstyle="-", color="blue", connectionstyle="arc3,rad=-0.3"
#             ),
#         )

#     showSave(fig)


class model:

    global show
    global techList
    global stockData
    
    accuracy = pd.DataFrame(columns=[">0", "==0", "<0"])
    accuracy.index.name = "Ticker"

    for ticker in tqdm(techList):

        df = yf.download(ticker, start="2012-01-01", end=datetime.now(), progress=False)

        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(f"Close Price History ({ticker})")
        plt.plot(df["Close"])
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Close Price USD ($)", fontsize=18)
        showSave(fig)

        # # Model Development

        # Create a new dataframe with only the 'Close column
        data = df.filter(["Close"])
        # Convert the dataframe to a numpy array
        dataset = data.values
        # Get the number of rows to train the model on
        trainingDataLen = int(
            np.ceil(len(dataset) * 0.80)
        )  # .95 leaves last .05 for test data
        
        
        # Scale the data

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaledData = scaler.fit_transform(dataset)
        
        
        # Create the training data set
        # Create the scaled training data set
        trainData = scaledData[
            0 : int(trainingDataLen), :
        ]  # shape is a column with (scaled) price data over time

        # Split the data into xTrain and yTrain data sets
        xTrain = []
        yTrain = []

        # xtrain is 60 periods of data, y train is 1 period after that
        for i in range(60, len(trainData)):
            xTrain.append(trainData[i - 60 : i, 0])
            yTrain.append(trainData[i, 0])

        # Convert the xTrain and yTrain to numpy arrays
        xTrain, yTrain = np.array(xTrain), np.array(yTrain)

        len(xTrain)
        print(f"\n\nlen(xTrain) {ticker}: {len(xTrain)}")

        # Reshape the data
        xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
        # xTrain.shape

        # Build the LSTM model
        model = Sequential()
        model.add(Input(shape=(xTrain.shape[1], 1)))
        model.add(LSTM(128, return_sequences=True, activation="relu"))
        model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
        model.add(LSTM(64, return_sequences=False, activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(25, activation="relu"))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer="adam", loss="mean_squared_error")

        TensorBoard = keras.callbacks.TensorBoard

        # Specify log directory within the current directory
        logDir = f"./logs/fit{ticker}"
        tensorBoardCallback = keras.callbacks.TensorBoard(
            log_dir=logDir, histogram_freq=1
        )

        print(f"\nNow Training {ticker}\n")

        newTrain = False
        
        if newTrain:
            # Train the model
            model.fit(
                xTrain,
                yTrain,
                batch_size=5,
                epochs=25,
                verbose=1,
                callbacks=[tensorBoardCallback],
            )  # Adjust epochs and batch size as needed

            model.save(f"./Model/LSTM{ticker}.keras")
            
        else:
            model = keras.models.load_model(f"./Model/LSTM{ticker}.keras")

        # Create the testing data set
        # Create a new array containing scaled values from index 1543 to 2002
        testData = scaledData[trainingDataLen - 60 :, :]
        # Create the data sets xTest and yTest
        xTest = []
        yTest = dataset[trainingDataLen:, :]
        for i in range(60, len(testData)):
            xTest.append(
                testData[i - 60 : i, 0]
            )
        
        # Convert the data to a numpy array
        xTest = np.array(xTest)


        # Reshape the data
        xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))
        
        

        print(f"\nNow Predicting {ticker}\n")
        # Get the models predicted price values
        predictions = model.predict(xTest)
        predictions = scaler.inverse_transform(predictions)
        
        # Get the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean(((predictions - yTest) ** 2)))
        rmse
        print(f"\nrmse {ticker}: {rmse}\n")

        # Plot the data
        train = data[:trainingDataLen]
        test = data[trainingDataLen:]
        test["Predictions"] = predictions
        test["Error (%)"] = [None] * len(test)
        for index, row in test.iterrows():

            test.loc[index, "Error (%)"] = (
                (row["Predictions"] - row["Close"]) / row["Close"] * 100
            )
        print(f"{ticker} Accuracy:\n{test["Error (%)"].describe()}")
        countGreater = sum(1 for num in test["Error (%)"] if num > 0)
        print(f"{ticker} Accuracy >0: {countGreater}")
        countEqual = sum(1 for num in test["Error (%)"] if num == 0)
        print(f"{ticker} Accuracy ==0: {countEqual}")
        countLess = sum(1 for num in test["Error (%)"] if num < 0)
        print(f"{ticker} Accuracy <0: {countLess}")
        
        accuracy.loc[ticker] = [countGreater, countEqual, countLess]
        # Visualize the data
        fig = plt.figure(figsize=(16, 6))
        fig.suptitle(f"Model {ticker}")
        plt.xlabel("Date", fontsize=18)
        plt.ylabel("Close Price USD ($)", fontsize=18)
        plt.plot(train["Close"])
        plt.plot(test[["Close", "Predictions"]])
        plt.legend(["Train", "Val", "Predictions"], loc="lower right")
        showSave(fig, True)

        # Calculate evaluation metrics
        mae = mean_absolute_error(yTest, predictions)
        mse = mean_squared_error(yTest, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(yTest, predictions)

        # Create a figure
        fig = plt.figure(figsize=(12, 6))

        # Set super title
        fig.suptitle(f"Model Evaluation and Predictions {ticker}", fontsize=16)

        # Plot true values vs. predictions
        # Scatter plot of true values vs. predictions
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(yTest, predictions, color="blue", label="True vs. Predictions")
        ax1.plot(yTest, yTest, color="red", linestyle="--", label="Perfect Predictions")
        ax1.set_title("True Values vs. Predictions")
        ax1.set_xlabel("True Values")
        ax1.set_ylabel("Predictions")
        ax1.legend()
        ax1.grid(True)

        # Bar chart of evaluation metrics
        # Create another subplot for metrics
        ax2 = fig.add_subplot(1, 2, 2)
        metrics = [
            "Mean Absolute Error",
            "Mean Squared Error",
            "Root Mean Squared Error",
            "R-squared Score",
        ]
        values = [mae, mse, rmse, r2]
        ax2.bar(metrics, values, color="skyblue")
        ax2.set_xlabel("Metrics")
        ax2.set_ylabel("Value")
        ax2.set_title("Model Evaluation Metrics")
        ax2.set_xticklabels(metrics, rotation=45, ha="right")
        ax2.grid(axis="y", linestyle="--", alpha=0.7)

        # Add R-squared value to the plot
        ax2.text(
            0.5,
            0.95,
            f"R-squared: {r2:.2f}",
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax2.transAxes,
            fontsize=10,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Adjust layout
        plt.tight_layout()

        # Show or save plot
        showSave(fig, True)

        fig = plt.figure(figsize=(10, 6))
        plt.suptitle(f"Analysis of Error Percentages {ticker}")

        plt.bar(test.index, test["Error (%)"], color="skyblue")
        plt.xlabel("Date")
        plt.ylabel("Error (%)")
        plt.title("Percent Error for Each Date")
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        

        plt.tight_layout()

        showSave(fig, True)

    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Accuracy Comparison", fontsize=16, fontweight='bold', y=0.95)
    accuracy.plot(kind="bar", ax=fig.add_subplot(111))
    plt.xlabel("Ticker")
    plt.ylabel("Count")
    plt.title("Accuracy Comparison of Tickers")
    plt.xticks(range(len(accuracy.index)), accuracy.index, rotation=0)
    plt.legend(title="Accuracy")
    plt.tight_layout()
    
    showSave(fig, True)
