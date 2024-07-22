# Guide https://pypi.org/project/yfinance/
# Guide https://algotrading101.com/learn/yahoo-finance-api-guide/
import yfinance as yf
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model, Sequential
import streamlit as st
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt


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

    def test(self):
        a = TestFun(self.history_data_file_path)


class TestFun():
    def __init__(self, history_data_file_path):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.time_step = 60  # 使用前60天的數據進行預測
        self.scaled_close_prices = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.history_data_file_path = history_data_file_path

        self.step1()
        self.step2()

    def step1(self):
        # 讀取CSV文件
        data = pd.read_csv(self.history_data_file_path)

        # 選擇'Close'
        close_prices = data[['Close']].values

        # 標準化數據
        self.scaled_close_prices = self.scaler.fit_transform(close_prices)

        # 創建訓練和測試數據集
        x, y = self.create_dataset(self.scaled_close_prices, self.time_step)

        # 拆分數據集為訓練和測試集
        train_size = int(len(x) * 0.8)
        test_size = len(x) - train_size
        self.x_train, self.x_test = x[0:train_size], x[train_size:len(x)]
        self.y_train, self.y_test = y[0:train_size], y[train_size:len(y)]

        # 將數據reshape為Transformer模型的輸入格式
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

    def step2(self):
        # 模型參數
        input_shape = (self.time_step, 1)  # 使用前60天的數據進行預測
        embed_dim = 64
        num_heads = 2
        ff_dim = 32

        # 構建模型
        transformer_model = self.build_transformer_model(input_shape, embed_dim, num_heads, ff_dim)

        # 編譯模型
        transformer_model.compile(optimizer='adam', loss='mean_squared_error')

        # 訓練模型
        transformer_model.fit(self.x_train, self.y_train, batch_size=32, epochs=10, validation_data=(self.x_test, self.y_test))

        # 使用模型進行預測
        train_predict = transformer_model.predict(self.x_train)
        test_predict = transformer_model.predict(self.x_test)

        # 反標準化數據
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)
        self.y_train = self.scaler.inverse_transform([self.y_train])
        self.y_test = self.scaler.inverse_transform([self.y_test])

        # 使用Streamlit顯示結果
        st.title('Stock Price Prediction using Transformer')
        st.subheader('AAPL Stock Price')

        # 繪製結果
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(self.scaler.inverse_transform(self.scaled_close_prices), label='Original Data')
        ax.plot(np.arange(self.time_step, len(train_predict) + self.time_step), train_predict, label='Training Prediction')
        ax.plot(
            np.arange(len(train_predict) + (2 * self.time_step), len(train_predict) + (2 * self.time_step) + len(test_predict)),
            test_predict, label='Testing Prediction')
        ax.legend()
        # 使用Streamlit顯示圖表
        st.pyplot(fig)

    @staticmethod
    def create_dataset(data, time_step=1):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            a = data[i:(i + time_step), 0]
            X.append(a)
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    # 構建Transformer模型
    def build_transformer_model(self, input_shape, embed_dim, num_heads, ff_dim):
        inputs = tf.keras.Input(shape=input_shape)
        transformer_block = self.TransformerEncoder(embed_dim, num_heads, ff_dim)
        x = transformer_block(inputs)
        x = Flatten()(x)
        x = Dense(20, activation="relu")(x)
        x = Dropout(0.1)(x)
        outputs = Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

    # 定義Transformer Encoder層
    class TransformerEncoder(tf.keras.layers.Layer):
        def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
            super(TestFun.TransformerEncoder, self).__init__()
            self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
            self.ffn = Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim)])
            self.layernorm1 = LayerNormalization(epsilon=1e-6)
            self.layernorm2 = LayerNormalization(epsilon=1e-6)
            self.dropout1 = Dropout(rate)
            self.dropout2 = Dropout(rate)

        def call(self, inputs, training):
            attn_output = self.att(inputs, inputs)
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output)
