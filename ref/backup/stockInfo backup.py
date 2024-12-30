# Guide https://pypi.org/project/yfinance/
# Guide https://algotrading101.com/learn/yahoo-finance-api-guide/
import yfinance as yf
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model, Sequential, load_model
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from .learner import ModelControl
# import matplotlib.pyplot as plt
from common import debug_msg

is_debug = True

class StockManager:
    def __init__(self, abs_path):
        self.history_data_folder_path = os.path.join(abs_path, "data")
        self.model_folder_path = os.path.join(abs_path, "model")
        self.__init_dir()

        # 讀取庫存的CSV名稱 集中成list管理
        self.stock_name_list = self.__init_load_stock_list()

        # 建立stock class 集中成list管理
        self.stock_class_list: list[Stock] = self.__init_create_stock_class()

        self.model_file_list = os.listdir(self.model_folder_path)
        self.model_file_list = [os.path.splitext(file)[0] if file.endswith('.pth') else file for file in self.model_file_list]

        self.msg = []
        self.is_action_successful = False

    def __init_dir(self):
        """檢查並建立基本資料夾"""
        if not os.path.exists(self.history_data_folder_path):
            os.mkdir(self.history_data_folder_path)
        if not os.path.exists(self.model_folder_path):
            os.mkdir(self.model_folder_path)

    def __init_load_stock_list(self):
        """讀取庫存的CSV名稱 集中成list管理"""
        file_list = os.listdir(self.history_data_folder_path)
        file_list_without_extension = [os.path.splitext(file)[0] if file.endswith('.csv') else file for file in file_list]
        return file_list_without_extension

    def __init_create_stock_class(self):
        """建立stock class 集中成list管理"""
        stock_class_list = []
        for stock_name in self.stock_name_list:
            stock_class_list.append(Stock(self.history_data_folder_path, self.model_folder_path, stock_name))
        return stock_class_list
    
    def __get_stock_index(self, stock_name: str) -> int:
        """
        取得stock_name在stock_name_list的index
        :param stock_name: str
        :return: index if stock_name in stock_name_list else -1
        """
        try:
            return self.stock_name_list.index(stock_name)
        except ValueError:
            return -1

    def create_stock_class(self, stock_name: str):
        """try create"""
        self.msg.clear()
        if stock_name in self.stock_name_list:
            self.is_action_successful = False
            debug_msg(is_debug, "This stock already exists in the list.")
            self.msg.append("This stock already exists in the list.")
            return

        # try downloading ticker
        try:
            stock = Stock(self.history_data_folder_path, self.model_folder_path, stock_name)

            if stock.download_ticker():
                stock.download_history_data()

                debug_msg(is_debug, f"{stock_name} Addition completed")
                self.msg.append(f"{stock_name} Addition completed")

                self.stock_class_list.append(stock)
                self.stock_name_list.append(stock_name)
                self.is_action_successful = True
            else:
                self.msg.append(f"{stock_name}: Stock name error. You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/.")
                self.is_action_successful = False

        except Exception as e:
            self.is_action_successful = False
            debug_msg(is_debug, e)
            self.msg.append(f"{stock_name}: {e}")
            return

    def update_history(self, **kwargs):
        self.msg.clear()

        # for streamlit
        if "stock_name" in kwargs:
            stock_index = self.__get_stock_index(kwargs["stock_name"])
        # for console
        elif "stock_index" in kwargs:
            stock_index = kwargs["stock_index"]
        else:
            return
        
        self.msg.append(self.stock_class_list[stock_index].download_history_data())
        self.is_action_successful = True
        return self.msg

    def update_all(self):
        """更新所有股票資訊"""
        self.msg.clear()
        if self.stock_class_list:
            for stock in self.stock_class_list:
                self.msg.append(stock.download_history_data())
            self.is_action_successful = True
        else:
            debug_msg(is_debug, "There is no data in the system.")
            self.msg.append("There is no data in the system.")
            self.is_action_successful = False

    # for streamlit
    def remove_stock(self, stock_name: str):
        try:
            self.msg.clear()
            stock_index = self.stock_name_list.index(stock_name)
            path = self.stock_class_list[stock_index].history_data_file_path
            os.remove(path)

            self.stock_name_list.remove(stock_name)
            self.stock_class_list.pop(stock_index)
            self.msg.append(f"{stock_name}: Remove completed.")

            self.is_action_successful = True
        except Exception as e:
            self.msg.append(f"{stock_name}: {e}")
            self.is_action_successful = False

    # for streamlit
    def get_stock_list(self):
        """顯示庫存的CSV名稱"""
        if self.stock_name_list:
            return self.stock_name_list
        else:
            debug_msg(is_debug, "No Data.")
            self.msg.append("No Data.")
            return False
    
    # for console
    def show_stock_list(self):
        """顯示庫存的CSV名稱"""
        if self.stock_name_list:
            for index, stock in enumerate(self.stock_name_list):
                print(f"{index:>3d}. {stock}")
        else:
            print("No Data.")

    # for console
    def show_history_data(self, stock_index: int):
        self.stock_class_list[stock_index].show_history_data()
        
    # for streamlit
    def get_company_info(self, stock_name):
        self.msg.clear()
        stock_index = self.__get_stock_index(stock_name)
        try:
            if self.stock_class_list[stock_index].get_company_info():
                info = self.stock_class_list[stock_index].company_info

                self.msg.append(info)
                debug_msg(is_debug, info)
                self.is_action_successful = True
                return info
            else:
                self.is_action_successful = False

        except Exception as e:
            self.msg.append(f"{stock_name}: {e}")
            self.is_action_successful = False

    # for console
    def show_company_info(self, stock_index: int):
        self.stock_class_list[stock_index].show_company_info()
    
    # for streamlit
    def get_analysis(self, stock_name: str):
        return self.show_prediction(self.__get_stock_index(stock_name), show_fig=False, show_go_fig=False)
    
    # for console
    def show_prediction(self, stock_index: int, model_file_path: str = None, show_fig: bool = None, show_go_fig: bool = None, creat_new_model: bool = False, retrain_model: bool = False):
        if model_file_path is None:
            model_file_path = self.stock_class_list[stock_index].model_file_path
        a = ModelControl(self.stock_class_list[stock_index].history_data_file_path, model_file_path, self.stock_name_list[stock_index], "Close")
        a.showFig = show_fig
        a.showGoFig = show_go_fig
        a.creat_new_model = creat_new_model
        a.retrain_model = retrain_model
        
        return a.fig

class Stock:
    def __init__(self, history_data_folder_path, model_folder_path, stock_name: str):
        self.stock_name = stock_name

        # 建立歷史資料檔案路徑(含副檔名)
        self.history_data_file_path = os.path.join(history_data_folder_path, self.stock_name + ".csv")

        # 建立模型檔案路徑(含副檔名)
        self.model_file_path = os.path.join(model_folder_path, self.stock_name + ".pth")

        # 初始化為None，待使用者輸入需求時再抓取
        self.ticker = None
        self.company_info = None
        self.history_data = None

    def download_ticker(self):
        """下載 ticker"""

        self.ticker = yf.Ticker(self.stock_name)

        # 股票名稱錯誤時，仍會返回一個dict，利用下列特徵確認股票名稱是否正確
        if 'previousClose' not in self.ticker.info:
            # raise AssertionError("Stock name error.")
            self.ticker = False
            return False
        else:
            return self.ticker

    def get_company_info(self):
        self.download_ticker()
        if self.ticker:
            self.company_info = self.ticker.info
        return self.ticker

    def show_company_info(self):
        """顯示 CompanyInfo"""
        if self.ticker is None:
            self.ticker = self.download_ticker()
            if self.ticker is False:
                return

        if self.company_info is None:
            self.company_info = self.ticker.info

        for key in self.company_info.keys():
            # debug_msg(is_debug, f"{key:30s} {self.company_info[key]}")
            print(f"{key:30s} {self.company_info[key]}")

    def show_history_data(self):
        """顯示 HistoryData"""
        if self.history_data is None:
            try:
                self.history_data = pd.read_csv(self.history_data_file_path)
                print(self.history_data)
            except Exception as e:
                debug_msg(is_debug, e)
                return e
        
    def download_history_data(self, period: str = 'max', interval: str = '1d'):
        """Download HistoryData"""

        self.ticker = self.download_ticker()

        try:
            if self.ticker:
                self.history_data = self.ticker.history(period=period, interval=interval)
                self.history_data['Date'] = pd.to_datetime(self.history_data.index).strftime('%Y-%m-%d')
                self.history_data.to_csv(self.history_data_file_path, index=False)
                debug_msg(is_debug, f"{self.stock_name}: Update completed.")
                return f"{self.stock_name}: Update completed."
            else:
                debug_msg(is_debug, "Stock name error. You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/")
                return f"{self.stock_name}: Stock name error. You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/."
        except Exception as e:
            return f"{self.stock_name}: {e}"
    
    def show_prediction(self):
        a = TestFun(self.history_data_file_path, self.model_file_path, self.stock_name)
        return a.show_prediction()

    def retrain_model(self):
        a = TestFun(self.history_data_file_path, self.model_file_path, self.stock_name)
        return a.retrain_model()
    
    def new_model(self):
        if os.path.exists(self.model_file_path):
            os.remove(self.model_file_path)
        self.retrain_model()


class TestFun():
    def __init__(self, history_data_file_path, model_file_path, stock_name):
        
        self.time_step = 120  # 使用前n天的數據進行預測
        self.scaled_close_prices = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.epochs = 10

        self.stock_name = stock_name
        self.data = pd.read_csv(history_data_file_path)
        self.scaler = MinMaxScaler(feature_range=(0, 1)).fit(self.data[['Close']])
        self.model_file_path = model_file_path

    def show_open_data(self):
        # 創建Plotly圖表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data["Date"], y=self.data["Close"], mode='lines', name='收盤價'))
        
        # 設置圖表佈局
        fig.update_layout(
            title=f"{self.stock_name} 收盤價",
            xaxis_title="日期",
            yaxis_title="收盤價 (USD)",
            xaxis_rangeslider_visible=True
        )
        
        # 使用st.plotly_chart顯示圖表
        st.plotly_chart(fig)
        
        return fig
    
    def show_prediction(self):
        self.step1_create_dataset()
        if not os.path.exists(self.model_file_path):
            model = self.step2_build_model()
        else:
            model = load_model(self.model_file_path, custom_objects={'TransformerEncoder': TestFun.TransformerEncoder})
        return self.step3(model)
    
    def retrain_model(self):
        """重新訓練模型"""
        self.step1_create_dataset()  # 創建數據集
        if not os.path.exists(self.model_file_path):
            model = self.step2_build_model()
        else:
            model = load_model(self.model_file_path, custom_objects={'TransformerEncoder': TestFun.TransformerEncoder})
            model.fit(self.x_train, self.y_train, batch_size=32, epochs=self.epochs, validation_data=(self.x_test, self.y_test))  # 進行訓練
        model.save(self.model_file_path)  # 保存模型

    def step1_create_dataset(self):
        data = self.data.copy()
        data = data[['Close']]

        # 標準化數據
        data['Close'] = self.scaler.fit_transform(data[['Close']])
        self.scaled_close_prices = data['Close']

        dataset = data.values

        # 創建訓練和測試數據集
        x, y = self.create_dataset(dataset, self.time_step)

        # 拆分數據集為訓練和測試集
        train_size = int(len(x) * 0.8)
        test_size = len(x) - train_size
        self.x_train, self.x_test = x[0:train_size], x[test_size:len(x)]
        self.y_train, self.y_test = y[0:train_size], y[test_size:len(y)]

        # 將數據reshape為Transformer模型的輸入格式
        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

    def step2_build_model(self):
        # 模型參數
        input_shape = (self.time_step, 1)  # 使用前n天的數據進行預測
        embed_dim = 64
        num_heads = 2
        ff_dim = 32

        # 構建模型
        transformer_model = self.build_transformer_model(input_shape, embed_dim, num_heads, ff_dim)

        # 編譯模型
        transformer_model.compile(optimizer='adam', loss='mean_squared_error')
        # 訓練模型
        transformer_model.fit(self.x_train, self.y_train, batch_size=32, epochs=self.epochs, validation_data=(self.x_test, self.y_test))

        # 保存模型
        transformer_model.save(self.model_file_path)

        return transformer_model

    def step3(self, transformer_model):
        # 使用模型進行預測
        train_predict = transformer_model.predict(self.x_train)
        test_predict = transformer_model.predict(self.x_test)

        # 反標準化數據
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)
        self.y_train = self.scaler.inverse_transform([self.y_train])
        self.y_test = self.scaler.inverse_transform([self.y_test])

        # 使用Streamlit顯示結果
        st.subheader('Stock Price Prediction using Transformer')
        st.subheader(f'{self.stock_name} Stock Price')

        # 繪製結果
        fig = go.Figure()
        # 需要調整
        # fig.add_trace(go.Scatter(x=np.arange(len(self.scaled_close_prices)), y=self.scaler.inverse_transform(self.scaled_close_prices).flatten(), mode='lines', name='原始數據'))
        fig.add_trace(go.Scatter(x=np.arange(self.time_step, len(train_predict) + self.time_step), y=train_predict.flatten(), mode='lines', name='Training Prediction'))
        fig.add_trace(go.Scatter(x=np.arange(len(train_predict) + (2 * self.time_step), len(train_predict) + (2 * self.time_step) + len(test_predict)), y=test_predict.flatten(), mode='lines', name='Testing Prediction'))
        
        # Set chart layout
        fig.update_layout(
            title=f"{self.stock_name}",
            xaxis_title="Date",
            yaxis_title="Closing Price (USD)",
            xaxis_rangeslider_visible=True
        )
        fig.show()
        return fig

    @staticmethod
    def create_dataset(data, time_step=1):
        x, y = [], []
        for i in range(len(data) - time_step - 1):
            x.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(x), np.array(y)

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
