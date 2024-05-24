# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

# For reading stock data from yahoo


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


import yfinance as yf


yf.pdr_override()

# For time stamps
from datetime import datetime


# The tech stocks we'll use for this analysis
techList = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]

# Set up End and Start times for data grab
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)

# Dictionary to store dataframes for each stock
stockData = {}

# Download data for each stock and store in the dictionary
for stock in techList:
    stockData[stock] = yf.download(stock, start, end)

# List of downloaded company dataframes
companyList = [stockData[stock] for stock in techList]
# Names of the corresponding companies
companyName = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON", "TESLA"]

# Add a column 'company_name' to each dataframe to identify the company
for company, comName in zip(companyList, companyName):
    company["company_name"] = comName

# Concatenate all dataframes into a single dataframe
df = pd.concat(companyList, axis=0)

# Display the concatenated dataframe
df
print(f"df :{df}")

# %%

# Describe TSLA dataframe
print(stockData["TSLA"].describe())


# %%
# Let's see a historical view of the closing price
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(companyList, 1):
    plt.subplot(3, 2, i)
    company["Adj Close"].plot()
    plt.ylabel("Adj Close")
    plt.xlabel(None)
    plt.title(f"Closing Price of {techList[i - 1]}")

plt.tight_layout()

# %%
# Now let's plot the total volume of stock being traded each day
plt.figure(figsize=(15, 10))
plt.subplots_adjust(top=1.25, bottom=1.2)

for i, company in enumerate(companyList, 1):
    plt.subplot(3, 2, i)
    company["Volume"].plot()
    plt.ylabel("Volume")
    plt.xlabel(None)
    plt.title(f"Sales Volume for {techList[i - 1]}")

plt.tight_layout()

# %%
ma_day = [10, 20, 50]

# Calculate moving averages and RSI for each company
for ma in ma_day:
    for company in companyList:
        column_name_ma = f"MA for {ma} days"
        company[column_name_ma] = company["Adj Close"].rolling(ma).mean()


# %%

# Calculate RSI for each company
rsiPeriod = 14  # RSI period
for company in companyList:
    delta = company["Adj Close"].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avgGain = gain.rolling(window=rsiPeriod, min_periods=1).mean()
    avgLoss = loss.rolling(window=rsiPeriod, min_periods=1).mean()

    rs = avgGain / avgLoss
    rsi = 100 - (100 / (1 + rs))

    company["RSI"] = rsi


# %%
companyName
print(f"company_name :{companyName}")

# %%
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(30, 20))
fig.suptitle("Stock Analysis")

# List of stock symbols
stock_symbols = ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"]


# Function to plot the data for a given company
def plot_company_data(ax, companyDf, stockSymbol):
    companyDf[
        ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days", "RSI"]
    ].plot(ax=ax)
    ax.set_title(f"Closing Price and Indicators for {stockSymbol}")
    ax.legend()


# Plot data for each company on separate graphs
for ax, companyDf, stockSymbol in zip(
    axes.flatten(), ["AAPL", "GOOG", "MSFT", "AMZN", "TSLA"], stock_symbols
):
    plot_company_data(ax, companyDf, stockSymbol)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()


# %%
ma_day = [10, 20, 50]

for ma in ma_day:
    for company in companyList:
        columnName = f"MA for {ma} days"
        company[columnName] = company["Adj Close"].rolling(ma).mean()

fig, axes = plt.subplots(nrows=3, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

stockData["AAPL"][
    ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
].plot(ax=axes[0, 0])
axes[0, 0].set_title("APPLE")

stockData["GOOG"][
    ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
].plot(ax=axes[0, 1])
axes[0, 1].set_title("GOOGLE")

stockData["MSFT"][
    ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
].plot(ax=axes[1, 0])
axes[1, 0].set_title("MICROSOFT")

stockData["'AMZN'"][
    ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
].plot(ax=axes[1, 1])
axes[1, 1].set_title("AMAZON")

stockData["TSLA"][
    ["Adj Close", "MA for 10 days", "MA for 20 days", "MA for 50 days"]
].plot(
    ax=axes[2, 0]
)  # Use a different subplot for TSLA
axes[2, 0].set_title("TESLA")

fig.tight_layout()

plt.show()


# %%
# We'll use pct_change to find the percent change for each day
for company in companyList:
    company["Daily Return"] = company["Adj Close"].pct_change()

# Then we'll plot the daily return percentage
fig, axes = plt.subplots(nrows=3, ncols=2)
fig.set_figheight(10)
fig.set_figwidth(15)

stockData["AAPL"]["Daily Return"].plot(
    ax=axes[0, 0], legend=True, linestyle="--", marker="o"
)
axes[0, 0].set_title("APPLE")

stockData["GOOG"]["Daily Return"].plot(
    ax=axes[0, 1], legend=True, linestyle="--", marker="o"
)
axes[0, 1].set_title("GOOGLE")

stockData["MSFT"]["Daily Return"].plot(
    ax=axes[1, 0], legend=True, linestyle="--", marker="o"
)
axes[1, 0].set_title("MICROSOFT")

stockData["AMZN"]["Daily Return"].plot(
    ax=axes[1, 1], legend=True, linestyle="--", marker="o"
)
axes[1, 1].set_title("AMAZON")

stockData["TSLA"]["Daily Return"].plot(
    ax=axes[2, 0], legend=True, linestyle="--", marker="o"
)
axes[2, 0].set_title("TESLA")

fig.tight_layout()

# %%
plt.figure(figsize=(12, 9))

for i, company in enumerate(companyList, 1):
    plt.subplot(3, 2, i)
    company["Daily Return"].hist(bins=50)
    plt.xlabel("Daily Return")
    plt.ylabel("Counts")
    plt.title(f"{companyName[i - 1]}")

plt.tight_layout()

# %%
# Fetch historical data for tech stocks
closingDf = yf.download(techList, start=start, end=end)["Adj Close"]

# Make a new tech returns DataFrame
techRets = closingDf.pct_change()
techRets.head()


# %%
sns.jointplot(x="GOOG", y="GOOG", data=techRets, kind="scatter", color="seagreen")

# %%
sns.jointplot(x="GOOG", y="MSFT", data=techRets, kind="scatter")

# %%
sns.pairplot(techRets, kind="reg")

# %%
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
sns.heatmap(techRets.corr(), annot=True, cmap="summer")
plt.title("Correlation of stock return")

plt.subplot(2, 2, 2)
sns.heatmap(closingDf.corr(), annot=True, cmap="summer")
plt.title("Correlation of stock closing price")

# %%
rets = techRets.dropna()

area = np.pi * 20

plt.figure(figsize=(10, 8))
plt.scatter(rets.mean(), rets.std(), s=area)
plt.xlabel("Expected return")
plt.ylabel("Risk")

for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label,
        xy=(x, y),
        xytext=(50, 50),
        textcoords="offset points",
        ha="right",
        va="bottom",
        arrowprops=dict(arrowstyle="-", color="blue", connectionstyle="arc3,rad=-0.3"),
    )

# %%
# Get the stock quote
df = yf.download("TSLA", start="2012-01-01", end=datetime.now())
# Show teh data
df
print(f"df :{df}")

# %%
plt.figure(figsize=(16, 6))
plt.title("Close Price History")
plt.plot(df["Close"])
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.show()

# %% [markdown]
# # Model Development

# %%
# Create a new dataframe with only the 'Close column
data = df.filter(["Close"])
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
trainingDataLen = int(np.ceil(len(dataset) * 0.95))

trainingDataLen
print(f"training_data_len :{trainingDataLen}")


# %%
# Scale the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
scaledData = scaler.fit_transform(dataset)

scaledData
print(f"scaled_data :{scaledData}")

# %%
# Create the training data set
# Create the scaled training data set
trainData = scaledData[0 : int(trainingDataLen), :]
# Split the data into x_train and y_train data sets
xTrain = []
yTrain = []

for i in range(60, len(trainData)):
    xTrain.append(trainData[i - 60 : i, 0])
    yTrain.append(trainData[i, 0])
    if i <= 61:
        print(xTrain)
        print(yTrain)
        print()

# Convert the x_train and y_train to numpy arrays
xTrain, yTrain = np.array(xTrain), np.array(yTrain)

# Reshape the data
xTrain = np.reshape(xTrain, (xTrain.shape[0], xTrain.shape[1], 1))
# x_train.shape

# %%
from tensorflow import keras

Sequential = keras.models.Sequential
Dense = keras.layers.Dense
LSTM = keras.layers.LSTM
Dropout = keras.layers.Dropout

# Build the LSTM model
model = Sequential()
model.add(
    LSTM(
        128, return_sequences=True, input_shape=(xTrain.shape[1], 1), activation="relu"
    )
)
model.add(Dropout(0.2))  # Dropout layer to prevent overfitting
model.add(LSTM(64, return_sequences=False, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(25, activation="relu"))
model.add(Dense(1))

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(
    xTrain, yTrain, batch_size=5, epochs=25
)  # Adjust epochs and batch size as needed


# %%
# Create the testing data set
# Create a new array containing scaled values from index 1543 to 2002
testData = scaledData[trainingDataLen - 60 :, :]
# Create the data sets x_test and y_test
xTest = []
yTest = dataset[trainingDataLen:, :]
for i in range(60, len(testData)):
    xTest.append(testData[i - 60 : i, 0])

# Convert the data to a numpy array
xTest = np.array(xTest)

# Reshape the data
xTest = np.reshape(xTest, (xTest.shape[0], xTest.shape[1], 1))

# Get the models predicted price values
predictions = model.predict(xTest)
predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - yTest) ** 2)))
rmse
print(f"rmse :{rmse}")

# %%
# Plot the data
train = data[:trainingDataLen]
valid = data[trainingDataLen:]
valid["Predictions"] = predictions
# Visualize the data
plt.figure(figsize=(16, 6))
plt.title("Model")
plt.xlabel("Date", fontsize=18)
plt.ylabel("Close Price USD ($)", fontsize=18)
plt.plot(train["Close"])
plt.plot(valid[["Close", "Predictions"]])
plt.legend(["Train", "Val", "Predictions"], loc="lower right")
plt.show()

# %%
