import pandas as pd
import numpy as np
import math
import json

import plotly.graph_objects as go
# import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('webagg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import os
MODE = os.getenv('MODE')
from common import log_stream

# https://ithelp.ithome.com.tw/articles/10361025
# https://medium.com/ching-i/transformer-attention-is-all-you-need-c7967f38af14
# https://medium.com/ching-i/pytorch-%E5%9F%BA%E6%9C%AC%E4%BB%8B%E7%B4%B9%E8%88%87%E6%95%99%E5%AD%B8-ac0e1ebfd7ec

class ModelControl:
    def __init__(self, **kwargs):
        self.stock_name = kwargs['stock_name']
        self.column = kwargs['column']
        self.using_data = kwargs.get('using_data', 1000)
        self.display_range = kwargs.get('display_range', self.using_data)
        self.dataFrame = pd.read_csv(kwargs['history_data_path'])[-self.using_data:]

        self.output_func = kwargs.get('output_func', self.do_nothing)
        self.test_MSE_func = kwargs.get('test_MSE_func', None)
        self.train_MSE_func = kwargs.get('train_MSE_func', None)

        self.src_model_path = kwargs.get('src_model_path', None)
        self.dst_model_path = kwargs.get('dst_model_path', None)
        self.train_size = kwargs.get('train_size', 0.8)
        self.test_size = 1 - self.train_size
        self.X_all_tensor = None

        # Hyperparameter
        self.window_width = kwargs.get('window_width', 60)
        self.num_epochs = kwargs.get('num_epochs', 100)
        
        # Hyperparameter
        self.num_layers = kwargs.get('num_layers', 2)
        self.nhead = kwargs.get('nhead', 4) # 4 會影響模型預測
        self.hidden_dim = kwargs.get('hidden_dim', 128) #128
        self.dropout = kwargs.get('dropout', 0.1) #0.1

        MODE = ['create', 'retrain', 'test', 'predict']
        self.operation_mode = kwargs.get('operation_mode', 'predict')
        if self.operation_mode not in MODE:
            raise ValueError(f"Invalid operation mode: {self.operation_mode}. Must be one of: {MODE}")
        self.predict_days = kwargs.get('predict_days', 0)

        self.showFig = kwargs.get('show_fig', None)
        self.showGoFig = kwargs.get('show_go_fig', None)
        self.display_epochs = kwargs.get('display_epochs', 10)

        # only for this class
        self.__scaler = MinMaxScaler(feature_range=(0, 1))
        self.__dataset = None
        self.__model = None
        self.__X_train_tensor, self.__y_train_tensor = None, None
        self.__X_test_tensor, self.__y_test_tensor = None, None

        self._fig = None
        self.__plot_date = None
        self.__plot_actual = None
        self.__plot_predicted = None
        self.__train_MSE = None
        self.__test_MSE = None

        self.__load_config()

    def __load_config(self):
        try:
            with open(self.src_model_path.replace('.pth', '.json'), 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.window_width = config['window_width']
            self.train_size = config['train_size']
            self.test_size = 1 - self.train_size
            print(f"The window_width setting {self.window_width} from config file changed to: {self.window_width}")
            print(f"The train_size setting {self.train_size} from config file changed to: {self.train_size}")
        except Exception as e:
            print("Error: ", e)
            print("No config file found. Using the input window_width: ", self.window_width)
            print("No config file found. Using the input train_size: ", self.train_size)

    @staticmethod
    def do_nothing(*args, **kwargs):
        pass

    def start(self):
        self.output_func("Preprocessing data...")
        log_stream(MODE, 'info', "Preprocessing data...")
        data = self.dataFrame[[self.column]].copy()
        # data[self.column] = data[self.column].pct_change() # 計算百分比變化
        # data[self.column] = data[self.column].fillna(1)  # 將NaN值轉換為1
        data[self.column] = self.__scaler.fit_transform(data[[self.column]])
        self.__X_train_tensor, self.__X_test_tensor, self.__y_train_tensor, self.__y_test_tensor = self.__step1_preprocess_data(data)

        # X_train 用於訓練
        # X_test 用於訓練
        # y_train 是用於比對預測偏差
        # y_test 是用於比對預測偏差
        
        self.output_func("Building/Loading model...")
        log_stream(MODE, 'info', "Building/Loading model...")
        self.__model = self.__step2_build_model()
        if self.operation_mode != 'predict':
            return

        self.output_func("Predicting...")
        log_stream(MODE, 'info', "Predicting...")
        self.__step3_predict(self.__model)
        log_stream(MODE, 'info', f"Train MSE: {self.__train_MSE}")
        log_stream(MODE, 'info', f"Test MSE: {self.__test_MSE}")

        if self.test_MSE_func is not None:
            self.test_MSE_func("Test MSE:", float(f"{self.__test_MSE:.4f}"))
        if self.train_MSE_func is not None:
            self.train_MSE_func("Train MSE:", float(f"{self.__train_MSE:.4f}"))
        self._fig = self.__step4_plot()
    
    def __step1_preprocess_data(self, data):
        def create_sequences(dataset, window_width=self.window_width):
            x, y = [], []
            for i in range(len(dataset) - window_width):
                x.append(dataset[i:(i + window_width), 0])
                y.append(dataset[i + window_width, 0])
            return np.array(x), np.array(y)

        # 定義 lookback 窗口大小
        self.__dataset = data.values
        x, y = create_sequences(self.__dataset, self.window_width)

        # x 是所有窗口的資料
        # y 是實際的數值

        # 劃分比例 訓練及測試
        train_size = int(len(x) * self.train_size)
        test_size = len(x) - train_size

        X_train = x[:train_size]
        X_test = x[train_size:]

        y_train = y[:train_size]
        y_test = y[train_size:]

        # 將數據轉換為 PyTorch 張量
        self.X_all_tensor = torch.from_numpy(x).float()
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).float()

        # 調整輸入形狀為 (seq_length, batch_size, feature_size)
        self.X_all_tensor = self.X_all_tensor.unsqueeze(-1).permute(1, 0, 2)
        X_train_tensor = X_train_tensor.unsqueeze(-1).permute(1, 0, 2)
        X_test_tensor = X_test_tensor.unsqueeze(-1).permute(1, 0, 2)

        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    def __step2_build_model(self):
        feature_size = 1
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        log_stream(MODE, 'info', f"You are using: {device}")
        
        # model.load_state_dict(torch.load(self.src_model_path))

        if self.operation_mode == 'create':
            model = ModelControl._TransformerTimeSeries(feature_size=feature_size, num_layers=self.num_layers, nhead=self.nhead, hidden_dim=self.hidden_dim, dropout=self.dropout).to(device)
        elif self.operation_mode == 'retrain':
            model = torch.load(self.src_model_path)
            model.feature_size = feature_size
            model.num_layers = self.num_layers
            model.nhead = self.nhead
            model.hidden_dim = self.hidden_dim
            model.dropout = self.dropout
        elif self.operation_mode == 'predict':
            model = torch.load(self.src_model_path)

        """
        舊方法
        if self.operation_mode != 'create':
            # model.load_state_dict(torch.load(self.src_model_path))
            model = torch.load(self.src_model_path)
        """

        if self.operation_mode == 'create' or self.operation_mode == 'retrain':
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # 轉換數據為 PyTorch 的 Tensor 並移動到設備
            self.__X_train_tensor = self.__X_train_tensor.to(device)
            self.__X_test_tensor = self.__X_test_tensor.to(device)

            self.__y_train_tensor = self.__y_train_tensor.to(device)
            self.__y_test_tensor = self.__y_test_tensor.to(device)

            # 訓練循環
            batch_size = self.__X_train_tensor.size(1)

            for epoch in range(self.num_epochs):
                model.train()
                optimizer.zero_grad()
                output = model(self.__X_train_tensor)
                loss = criterion(output.squeeze(), self.__y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % self.display_epochs == 0 or epoch == self.num_epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        test_output = model(self.__X_test_tensor)
                        test_loss = criterion(test_output.squeeze(), self.__y_test_tensor)
                    self.output_func(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
                    log_stream(MODE, 'info', f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

            # torch.save(model.state_dict(), self.dst_model_path)  # 儲存模型權重
            torch.save(model, self.dst_model_path) # 儲存模型
            
            with open(self.dst_model_path.replace('.pth', '.json'), 'w', encoding='utf-8') as f:
                config = {"window_width": self.window_width, 'train_size': self.train_size}
                json.dump(config, f)

        return model

    def __step3_predict(self, model):
        # 預測並反歸一化
        model.eval()
        with torch.no_grad():
            train_predict = model(self.__X_train_tensor).cpu().numpy()
            test_predict = model(self.__X_test_tensor).cpu().numpy()
        if self.predict_days > 0:
            self.new_days_predict = self.__predict_future(model)

        # 反歸一化
        train_predict = self.__scaler.inverse_transform(train_predict)
        y_train_actual = self.__scaler.inverse_transform(self.__y_train_tensor.cpu().numpy().reshape(-1, 1))
        test_predict = self.__scaler.inverse_transform(test_predict)
        y_test_actual = self.__scaler.inverse_transform(self.__y_test_tensor.cpu().numpy().reshape(-1, 1))

        if self.predict_days > 0:
            self.new_days_predict = self.__scaler.inverse_transform(self.new_days_predict)

        # 構建完整的時間序列
        predicted = np.concatenate((train_predict, test_predict), axis=0)
        actual = self.__scaler.inverse_transform(self.__dataset[self.window_width:])
        date = self.dataFrame['Date'].values[-len(actual):]

        self.__plot_date = date
        self.__plot_actual = actual
        self.__plot_predicted = predicted

        train_score = mean_squared_error(y_train_actual, train_predict)
        test_score = mean_squared_error(y_test_actual, test_predict)

        self.__train_MSE = train_score
        self.__test_MSE = test_score

    def __step4_plot(self):
        # calculate the display range
        if self.display_range > self.__plot_actual.shape[0] + self.__plot_predicted.shape[0] + self.predict_days:
            display_range = self.__plot_actual.shape[0] + self.__plot_predicted.shape[0] + self.predict_days
        else:
            display_range = self.display_range

        date = self.__append_date(self.__plot_date[- display_range + self.predict_days:], self.predict_days)
        actual = self.__plot_actual.flatten()[- display_range + self.predict_days:]
        # predicted = predicted.flatten()[- display_range - self.predict_days:]
        if self.predict_days > 0:
            predicted = np.concatenate((self.__plot_predicted[- display_range + self.predict_days:], self.new_days_predict), axis=0).flatten()
        else:
            predicted = self.__plot_predicted.flatten()[- display_range:]

        # 顯示圖表 for console
        if self.showFig is None:
            # fig.show()
            ans = input("Do you want to show the plot? (y/n)")
            if ans.lower() == "y":
                self.__show_plotly(date, actual, predicted)
        elif self.showFig is True:
            self.__show_plotly(date, actual, predicted)

        fig = self.__get_go_fig(date, actual, predicted)
        if self.showGoFig is None:
            ans = input("Do you want to show the fig? (y/n)")
            if ans.lower() == "y":
                fig.show()
        elif self.showGoFig is True:
            fig.show()
        
        return fig
    
    def __predict_future(self, model):
        """
        with torch.no_grad():
            new_predict_data = model(self.__X_test_tensor, predict_days=self.predict_days).cpu().numpy()
        """
        """
        predicted_prices = []
        for _ in range(30): # 預測未來 30 天
            next_price = model.predict(input_data) # 模型預測
            predicted_prices.append(next_price[0, -1, 0]) # 儲存最新預測值

            # 更新輸入數據：移除最舊數據，添加新預測值
            new_input = np.append(input_data[:, 1:, :], [[next_price[0, -1, 0]]], axis=1)
            input_data = np.expand_dims(new_input, axis=0)
        """
        src = self.X_all_tensor.clone()
        predicted_prices = []
        with torch.no_grad():
            for _ in range(self.predict_days):
                next_price = model(src).cpu().numpy()
                predicted_prices.append(next_price[-1, 0])
                new_input = torch.tensor([[[next_price[-1, 0]]]])
                src = torch.cat((src[:, 1:, :], new_input.repeat(self.window_width, 1, 1)), dim=1)
        predicted_prices = torch.tensor(predicted_prices).unsqueeze(1)
        return predicted_prices
    
    def __show_plotly(self, date, actual, predicted):
        plt.figure(figsize=(12,6))
        plt.plot(date, actual, label='Actual Price', color='blue')
        plt.plot(date, predicted, label='Predicted Price', color='red')
        plt.title(f'{self.stock_name} {self.column} Prediction - Transformer Model')
        plt.xlabel('Time')
        plt.ylabel(f'{self.column} (USD)')

        plt.xticks(ticks=date, labels=date, rotation=45)  # 每隔5個顯示一個時間標籤
        plt.legend()
        plt.show()

    def __append_date(self, date: np.array, days):
        import datetime
        date = date.copy()
        format = '%Y-%m-%d'
        try:
            last_date = datetime.datetime.strptime(date[-1], format)
        except:
            format = "%Y/%m/%d"
            last_date = datetime.datetime.strptime(date[-1], format)
        added_days = 0
        
        while added_days < days:
            last_date += datetime.timedelta(days=1)
            # 檢查是否為工作日（週一到週五）
            if last_date.weekday() < 5:  # 0-4 代表週一到週五
                date = np.append(date, last_date.strftime(format))
                added_days += 1
                
        return date

    def __get_go_fig(self, date, actual, predicted):
        # 創建圖表 for streamlit
        fig = go.Figure()
        # 添加實際價格的線
        fig.add_trace(go.Scatter(x=date, y=actual, marker=dict(color='blue'), mode='lines+markers', name='Actual Price'))
        # 添加預測價格的線
        fig.add_trace(go.Scatter(x=date, y=predicted, marker=dict(color='red'), mode='lines+markers', name='Predicted Price'))

        # 更新圖表的標題和軸標籤
        fig.update_layout(
            title=f"{self.stock_name} {self.column}",
            xaxis_title="Date",
            yaxis_title=f"{self.column} (USD)",
            xaxis_rangeslider_visible=True
        )

        # 設置 x 軸的刻度顯示
        # fig.update_xaxes(dtick='D10', tickformat='%Y-%m-%d', tickangle=45)  # 每n天顯示一次，並旋轉標籤

        return fig

    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(ModelControl._PositionalEncoding, self).__init__()
            # 建立一個位置編碼矩陣
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)  # 偶數位置
            pe[:, 1::2] = torch.cos(position * div_term)  # 奇數位置
            pe = pe.unsqueeze(1)
            self.register_buffer('pe', pe)

        def forward(self, x):
            # x 的形狀為 (seq_length, batch_size, d_model)
            x = x + self.pe[:x.size(0)]
            return x

    class _TransformerTimeSeries(nn.Module):
        def __init__(self, feature_size=1, num_layers=2, nhead=4, hidden_dim=128, dropout=0.1):
            super(ModelControl._TransformerTimeSeries, self).__init__()
            self.model_type = 'Transformer'
            self.input_linear = nn.Linear(feature_size, hidden_dim)  # 新增的線性層
            self.pos_encoder = ModelControl._PositionalEncoding(d_model=hidden_dim)
            encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
            self.decoder = nn.Linear(hidden_dim, 1)
            self.hidden_dim = hidden_dim

        def forward(self, src, predict_days=0):
            src = src.clone()

            while predict_days > -1:
                # src 形狀: (seq_length, batch_size, feature_size)
                src2 = self.input_linear(src)  # 將輸入映射到 hidden_dim 維度
                src2 = self.pos_encoder(src2)  # Positional Encoding 就是位置編碼，目的是為了讓模型考慮位置之間的順序
                output_org = self.transformer_encoder(src2)
                # 取最後一個時間步的輸出
                output = self.decoder(output_org[-1, :, :])
                # src = torch.cat((src[1:, :, :], output.unsqueeze(0)), dim=0)

                predict_days -= 1

            return output
        
            # 預測未來5日
            future_predictions = []
            output = output_org
            for _ in range(predict_days):  # 預測未來5日
                print("loop: ", _)
                output = self.decoder(output[-1, :, :]).unsqueeze(0)  # 將最後的輸出轉換為形狀 (1, batch_size, 1)
                print("decoder output: ", output.shape)
                print("src_for_prediction: ", src_for_prediction.shape)
                future_predictions.append(output)
                src_for_prediction = torch.cat((src_for_prediction, output), dim=0)  # 將預測的輸出添加到輸入序列中
                src_for_prediction = src_for_prediction[-60:]  # 保持序列長度不變

            return torch.cat(future_predictions, dim=0)  # 返回未來5日的預測


if __name__ == '__main__':
    test = ModelControl('Close')
    test.start()
