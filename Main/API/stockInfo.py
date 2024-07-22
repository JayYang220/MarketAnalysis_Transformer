# Guide https://pypi.org/project/yfinance/
# Guide https://algotrading101.com/learn/yahoo-finance-api-guide/
import yfinance as yf
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Import the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, TimeDistributed
from keras.callbacks import EarlyStopping


class StockManager:
    __slots__ = ["history_data_folder_path", "stock_name_list", "stock_class_list"]

    def __init__(self, abs_path):
        self.history_data_folder_path = os.path.join(abs_path, "Data")

        # 讀取庫存的CSV名稱 集中成list管理
        self.stock_name_list = self.__init_load_stock_list()

        # 建立stock class 集中成list管理
        self.stock_class_list = self.__init_create_stock_class()

    def __init_load_stock_list(self):
        """讀取庫存的CSV名稱 集中成list管理"""
        if os.path.exists(self.history_data_folder_path) is False:
            # 從未下載過任何資料時
            os.mkdir(self.history_data_folder_path)
            return []
        else:
            # 讀取並返回
            file_list = os.listdir(self.history_data_folder_path)
            file_list_without_extension = [os.path.splitext(file)[0] if file.endswith('.csv') else file for file in file_list]
            return file_list_without_extension

    def __init_create_stock_class(self):
        """建立stock class 集中成list管理"""
        stock_class_list = []
        for stock_name in self.stock_name_list:
            stock_class_list.append(Stock(self.history_data_folder_path, stock_name))
        return stock_class_list

    def create_stock_class(self, stock_name: str):
        """try create"""
        if stock_name in self.stock_name_list:
            print("This stock already exists in the list.")
            return

        # try downloading ticker
        try:
            stock = Stock(self.history_data_folder_path, stock_name)

            stock.download_ticker()
            stock.download_history_data()

            self.stock_class_list.append(stock)
            self.stock_name_list.append(stock_name)
            print(f"{stock_name} Addition completed")

        except Exception as e:
            print(e)
            print("You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/")
            return

    def update_all(self):
        """更新所有股票資訊"""
        if self.stock_class_list:
            for stock in self.stock_class_list:
                stock.download_history_data()
        else:
            print("There is no data in the system.")

    def show_stock_list(self):
        """顯示庫存的CSV名稱"""
        if self.stock_name_list:
            for index, stock in enumerate(self.stock_name_list):
                print(f"{index:>3d}. {stock}")
        else:
            print("No Data.")


class Stock:
    __slots__ = ["stock_name", "history_data_file_path", "ticker", "company_info", "history_data"]

    def __init__(self, history_data_folder_path, stock_name: str):
        self.stock_name = stock_name
        self.history_data_file_path = os.path.join(history_data_folder_path, self.stock_name + ".csv")

        # 初始化為None，待使用者輸入需求時再抓取
        self.ticker = None
        self.company_info = None
        self.history_data = None

    def download_ticker(self):
        """下載 ticker"""
        try:
            self.ticker = yf.Ticker(self.stock_name)

            # 股票名稱錯誤時，仍會返回一個dict，利用下列特徵確認股票名稱是否正確
            if 'previousClose' not in self.ticker.info:
                print("Stock name error.")
                # raise AssertionError("Stock name error.")
                return False
            else:
                return self.ticker

        except Exception as e:
            print("Error:", e)
            return False

    def show_company_info(self):
        """顯示 CompanyInfo"""
        if self.ticker is None:
            self.ticker = self.download_ticker()
            if self.ticker is False:
                return

        if self.company_info is None:
            self.company_info = self.ticker.info

        for key in self.company_info.keys():
            print(f"{key:30s} {self.company_info[key]}")

    def show_history_data(self):
        """顯示 HistoryData"""
        if self.history_data is None:
            self.load_history()
        print(self.history_data)

    def load_history(self):
        """讀取 HistoryData"""
        try:
            self.history_data = pd.read_csv(self.history_data_file_path)
        except Exception as e:
            print(e)

    def download_history_data(self, period: str = 'max', interval: str = '1d'):
        """Download HistoryData"""
        # 舊方法
        # import pandas_datareader.data
        # yf.pdr_override()
        # start_date = dt.datetime(1900, 1, 10)
        # end_date = dt.datetime(2100, 3, 18)
        # self.history = pandas_datareader.data.get_data_yahoo(self.stockName, start_date, end_date)

        self.ticker = self.download_ticker()

        if self.ticker:
            self.history_data = self.ticker.history(period=period, interval=interval)
            self.history_data.to_csv(self.history_data_file_path)
            print(f"{self.stock_name} Update completed")
        else:
            # 錯誤資訊會於self.download_ticker()顯示
            pass

    def lstm_function(self):
        """建立並繪圖"""
        AnalysisStock(self.stock_name, self.history_data_file_path)

    def test(self):
        # https://ithelp.ithome.com.tw/articles/10206312?authuser=1

        # 取消Alarm
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        datasetLen = 7306  # 設定訓練資料夾總數
        datatrainPre = 1200  # 取最近n日做為訓練用資料
        datasetReal = 20  # 設定最近n日的data作為真實資料
        timeStep = 60  # 設定多少區間作為時間步長

        try:
            # 讀取訓練集
            dataset_train = pd.read_csv(self.history_data_file_path)

            # 讀取所有行+第2列
            # .values是將DataFrame 轉換為 numpy array，也可以不用，因為後MinMaxScaler會自動轉換成numpy array再處理
            # 必須使用[:,1:2]，若用[:,1]只會得到1維array
            training_set = dataset_train.iloc[:, 1:2].values

            # 設定縮放range
            sc = MinMaxScaler(feature_range=(0, 1))
            # 進行縮放，返回2維 numpy array
            training_set_scaled = sc.fit_transform(training_set)

            X_train = []  # 預測點的前 60 天的資料
            y_train = []  # 預測點

            offset = datasetLen - datatrainPre  # 訓練資料集起始點偏移
            for i in range(timeStep + offset, datasetLen):
                # 將60為一個區間，step1，逐漸將所有data加入X_train
                X_train.append(training_set_scaled[i - timeStep:i, 0])
                # 將60為一個區間，step1，逐漸將所有data加入y_train
                y_train.append(training_set_scaled[i, 0])

            # 轉成numpy array的格式，以利輸入 RNN
            X_train, y_train = np.array(X_train), np.array(y_train)

            # 將X_train重整為3維array
            # np.reshape X_train.shape[0]=第1個維度大小 (樣本總數), X_train.shape[1]=第2個維度大小 (時間步長)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # Initialising the RNN
            regressor = Sequential()

            # Adding the first LSTM layer and some Dropout regularisation
            # units是神經元數量, return_sequences=True則是為了傳遞到下一層
            # Dropout是指訓練過程隨機將部分輸出設為0，以防止過擬合，增強泛化能力
            regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            regressor.add(Dropout(0.2))

            # Adding a second LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.2))

            # Adding a third LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units=50, return_sequences=True))
            regressor.add(Dropout(0.2))

            # Adding a fourth LSTM layer and some Dropout regularisation
            regressor.add(LSTM(units=50))
            regressor.add(Dropout(0.2))

            # Adding the output layer
            # 輸出神經元，通常只有1個
            regressor.add(Dense(units=1))

            # Compiling
            # optimizer優化器
            # loss損失函數
            regressor.compile(optimizer='adam', loss='mean_squared_error')

            # 進行訓練
            # epochs是指對整個訓練的完整跌代，增加可提高訓練精度但也可能導致過擬合
            # batch_size是指一次處理的樣本數量，批量大可加快訓練速度，批量小可加快收斂速度，但較不穩定
            regressor.fit(X_train, y_train, epochs=100, batch_size=32)

            # 製作真實資料
            dataset_real = dataset_train.tail(datasetReal)
            dataset_real = pd.concat([dataset_train.iloc[:1], dataset_real])
            real_stock_price = dataset_real.iloc[:, 1:2].values

            # 縱向合併
            dataset_total = pd.concat((dataset_train['Open'], dataset_real['Open']), axis=0)

            inputs = dataset_total[len(dataset_total) - len(dataset_real) - 60:].values
            inputs = inputs.reshape(-1, 1)
            inputs = sc.transform(inputs)  # Feature Scaling

            X_test = []
            for i in range(timeStep, timeStep + datasetReal):  # timeStep和之前一樣, timeStep+真實資料集
                X_test.append(inputs[i - timeStep:i, 0])
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape 成 3-dimension

            predicted_stock_price = regressor.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)  # to get the original scale

            # Visualising the results
            plt.plot(real_stock_price, color='red', label=f'Real {self.stock_name} Stock Price')  # 紅線表示真實股價
            plt.plot(predicted_stock_price, color='blue', label=f'Predicted {self.stock_name} Stock Price')  # 藍線表示預測股價
            plt.title(f'{self.stock_name} Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel(f'{self.stock_name} Stock Price')
            plt.legend()
            plt.show()

        except Exception as e:
            raise e

    def test2(self):
        # https://daniel820710.medium.com/%E5%88%A9%E7%94%A8keras%E5%BB%BA%E6%A7%8Blstm%E6%A8%A1%E5%9E%8B-%E4%BB%A5stock-prediction-%E7%82%BA%E4%BE%8B-1-67456e0a0b
        # Read
        train = pd.read_csv(self.history_data_file_path)

        # Augment Features
        # 刪除時區
        train["Date"] = train["Date"].str.replace(r'-\d{2}:\d{2}$', '', regex=True)
        train["Date"] = pd.to_datetime(train["Date"])
        train["year"] = train["Date"].dt.year
        train["month"] = train["Date"].dt.month
        train["date"] = train["Date"].dt.day
        train["day"] = train["Date"].dt.dayofweek

        # Normalization
        train = train.drop(["Date"], axis=1)
        train = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))


        # Build Training Data
        pastDay = 30
        futureDay = 5

        X_train, Y_train = [], []
        for i in range(train.shape[0] - futureDay - pastDay):
            X_train.append(np.array(train.iloc[i:i + pastDay]))
            Y_train.append(np.array(train.iloc[i + pastDay:i + pastDay + futureDay]["Close"]))

        # 資料亂序
        X, Y = np.array(X_train), np.array(Y_train)
        np.random.seed(10)
        randomList = np.arange(X.shape[0])
        np.random.shuffle(randomList)
        X, Y = X[randomList], Y[randomList]


        # Training data & Validation data
        rate = 0.1
        X_train = X[int(X.shape[0] * rate):]
        Y_train = Y[int(Y.shape[0] * rate):]
        X_val = X[:int(X.shape[0] * rate)]
        Y_val = Y[:int(Y.shape[0] * rate)]

        # from 2 dimension to 3 dimension
        Y_train = Y_train[:, np.newaxis]
        Y_val = Y_val[:, np.newaxis]

        model = self.test2_buildOneToOneModel(X_train.shape)
        callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
        model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])

    def test2_buildOneToOneModel(self, shape):
        model = Sequential()
        model.add(LSTM(10, input_length=shape[1], input_dim=shape[2], return_sequences=True))
        # output shape: (1, 1)
        model.add(TimeDistributed(Dense(1)))  # or use model.add(Dense(1))
        model.compile(loss="mse", optimizer="adam")
        model.summary()
        return model


class AnalysisStock:
    # https://ithelp.ithome.com.tw/articles/10206312?authuser=1
    def __init__(self, stock_name, history_data_file_path):
        # 取消Alarm
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self.stock_name = stock_name
        self.history_data_file_path = history_data_file_path

        self.dataset_len = self.__init_get_dataset_len()  # 取得訓練資料集，資料總數
        self.data_train_pre_len = 1200  # 取最近n日做為訓練用資料
        self.dataset_real_len = 20  # 設定最近n日的data作為真實資料
        self.time_step_len = 60  # 設定多少區間作為時間步長

        self.train_dataset = self.__init_get_train_dataset()  # 讀取訓練集
        self.sc = MinMaxScaler(feature_range=(0, 1))  # 設定縮放range

        self.step_control()

    def __init_get_dataset_len(self):
        """返回資料集資料總數"""
        return len(pd.read_csv(self.history_data_file_path))

    def __init_get_train_dataset(self):
        try:
            train_dataset = pd.read_csv(self.history_data_file_path)
            return train_dataset
        except Exception as e:
            print(e)

    def step_control(self):
        try:
            self.step1_prepare_data()
            self.step2_training()
            self.step3_output()
        except Exception as e:
            print(e)

    def step1_prepare_data(self):
        # 讀取所有行+第2列
        # .values是將DataFrame 轉換為 numpy array，也可以不用，因為後MinMaxScaler會自動轉換成numpy array再處理
        # 必須使用[:,1:2]，若用[:,1]只會得到1維array
        training_dataset_np = self.train_dataset.iloc[:, 1:2].values

        # 進行縮放，返回2維 numpy array
        training_dataset_scaled = self.sc.fit_transform(training_dataset_np)

        x_train = []  # 預測點的前 60 天的資料
        self.y_train = []  # 預測點

        offset = self.dataset_len - self.data_train_pre_len  # 訓練資料集起始點偏移
        for i in range(self.time_step_len + offset, self.dataset_len):
            # 將60為一個區間，step1，逐漸將所有data加入X_train
            x_train.append(training_dataset_scaled[i - self.time_step_len:i, 0])
            # 將60為一個區間，step1，逐漸將所有data加入y_train
            self.y_train.append(training_dataset_scaled[i, 0])

        # 轉成numpy array的格式，以利輸入 RNN
        self.x_train, self.y_train = np.array(x_train), np.array(self.y_train)

        # 將X_train重整為3維array
        # np.reshape X_train.shape[0]=第1個維度大小 (樣本總數), X_train.shape[1]=第2個維度大小 (時間步長)
        self.x_train = np.reshape(self.x_train, (self.x_train.shape[0], self.x_train.shape[1], 1))

    def step2_training(self):
        # Initialising the RNN
        self.regressor = Sequential()

        # Adding the first LSTM layer and some Dropout regularisation
        # units是神經元數量, return_sequences=True則是為了傳遞到下一層
        # Dropout是指訓練過程隨機將部分輸出設為0，以防止過擬合，增強泛化能力
        self.regressor.add(LSTM(units=50, return_sequences=True, input_shape=(self.x_train.shape[1], 1)))
        self.regressor.add(Dropout(0.2))

        # Adding a second LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.2))

        # Adding a third LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50, return_sequences=True))
        self.regressor.add(Dropout(0.2))

        # Adding a fourth LSTM layer and some Dropout regularisation
        self.regressor.add(LSTM(units=50))
        self.regressor.add(Dropout(0.2))

        # Adding the output layer
        # 輸出神經元，通常只有1個
        self.regressor.add(Dense(units=1))

        # Compiling
        # optimizer優化器
        # loss損失函數
        self.regressor.compile(optimizer='adam', loss='mean_squared_error')

        # 進行訓練
        # epochs是指對整個訓練的完整跌代，增加可提高訓練精度但也可能導致過擬合
        # batch_size是指一次處理的樣本數量，批量大可加快訓練速度，批量小可加快收斂速度，但較不穩定
        self.regressor.fit(self.x_train, self.y_train, epochs=100, batch_size=32)

    def step3_output(self):
        # 製作真實資料
        dataset_real_df = self.train_dataset.tail(self.dataset_real_len)    # 返回最後n行，作為真實資料
        stock_price_real_np = dataset_real_df.iloc[:, 1:2].values

        # 縱向合併
        dataset_total_df = pd.concat((self.train_dataset['Open'], dataset_real_df['Open']), axis=0)

        inputs = dataset_total_df[len(dataset_total_df) - len(dataset_real_df) - 60:].values.reshape(-1, 1)
        inputs = self.sc.transform(inputs)  # Feature Scaling

        X_test = []
        for i in range(self.time_step_len, self.time_step_len + self.dataset_real_len):  # timeStep和之前一樣, timeStep+真實資料集
            X_test.append(inputs[i - self.time_step_len:i, 0])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # Reshape 成 3-dimension

        predicted_stock_price = self.regressor.predict(X_test)
        predicted_stock_price = self.sc.inverse_transform(predicted_stock_price)  # to get the original scale

        # Visualising the results
        plt.plot(stock_price_real_np, color='red', label=f'Real {self.stock_name} Stock Price')  # 紅線表示真實股價
        plt.plot(predicted_stock_price, color='blue', label=f'Predicted {self.stock_name} Stock Price')  # 藍線表示預測股價
        plt.title(f'{self.stock_name} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(f'{self.stock_name} Stock Price')
        plt.legend()
        plt.show()


# 測試用
if __name__ == "__main__":
    stockName = "2330.TW"
    ticker = yf.Ticker(stockName)
    print(ticker.info)
