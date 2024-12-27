import pandas as pd
import numpy as np
import math
import os

import plotly.graph_objects as go
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



# https://ithelp.ithome.com.tw/articles/10361025
class TestTransformer:
    def __init__(self, history_data_file_path, model_file_path, stock_name, column):
        self.stock_name = stock_name
        self.column = column

        self.use_data = 1000 # 使用最後n筆資料量
        self.dataFrame = pd.read_csv(history_data_file_path)[-self.use_data:]
        self.module_path = model_file_path
        self.train_size = 0.8
        self.test_size = 1 - self.train_size
        
        self.window_width = 60 #60
        self.num_epochs = 100 #100

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = None
        self.model = None
        self.X_train_tensor, self.y_train_tensor = None, None
        self.X_test_tensor, self.y_test_tensor = None, None

        self.creat_new_model = False
        self.retrain_model = False
        self.showFig = False

    def step_control(self):
        print("Preprocessing data...")
        data = self.dataFrame[[self.column]].copy()
        
        # data[self.column] = data[self.column].pct_change() # 計算百分比變化
        # data[self.column] = data[self.column].fillna(1)  # 將NaN值轉換為1
        data[self.column] = self.scaler.fit_transform(data[[self.column]])
        self.X_train_tensor, self.y_train_tensor, self.X_test_tensor, self.y_test_tensor = self.__step1_preprocess_data(data)

        print("Building/Loading model...")
        if os.path.exists(self.module_path):
            if self.creat_new_model:
                ans = input("Model file already exists. Do you want to overwrite it? (y/n)")
                if ans.lower() != "y":
                    return
        elif self.retrain_model:
                print("Model file not found. Please create a new model first.")
                return
        else:
            self.creat_new_model = True
        
        self.model = self.__step2_build_model()

        print("Predicting...")
        fig = self.__step3_predict(self.model)
        return fig
    
    def __step1_preprocess_data(self, data):
        def create_sequences(dataset, window_width=self.window_width):
            x, y = [], []
            for i in range(len(dataset) - window_width):
                x.append(dataset[i:(i + window_width), 0])
                y.append(dataset[i + window_width, 0])
            return np.array(x), np.array(y)

        # 定義 lookback 窗口大小
        self.dataset = data.values
        x, y = create_sequences(self.dataset, self.window_width)

        # 劃分比例 訓練及測試
        train_size = int(len(x) * self.train_size)
        test_size = len(x) - train_size

        X_train = x[:train_size]
        y_train = y[:train_size]
        X_test = x[train_size:]
        y_test = y[train_size:]

        # 將數據轉換為 PyTorch 張量
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).float()

        # 調整輸入形狀為 (seq_length, batch_size, feature_size)
        print(X_train_tensor.shape)
        X_train_tensor = X_train_tensor.unsqueeze(-1).permute(1, 0, 2)
        print(X_train_tensor.shape)
        X_test_tensor = X_test_tensor.unsqueeze(-1).permute(1, 0, 2)

        return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
    
    def __step2_build_model(self):
        feature_size = 1
        num_layers = 2 # 2
        nhead = 4 # 4
        hidden_dim = 128 #128
        dropout = 0.1
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"You are using: {device}")
        
        model = TestTransformer._TransformerTimeSeries(feature_size=feature_size, num_layers=num_layers, nhead=nhead, hidden_dim=hidden_dim, dropout=dropout).to(device)
        if not self.creat_new_model:
            model.load_state_dict(torch.load(self.module_path))

        if self.creat_new_model or self.retrain_model:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # 轉換數據為 PyTorch 的 Tensor 並移動到設備
            self.X_train_tensor = self.X_train_tensor.to(device)
            self.y_train_tensor = self.y_train_tensor.to(device)
            self.X_test_tensor = self.X_test_tensor.to(device)
            self.y_test_tensor = self.y_test_tensor.to(device)

            # 訓練循環
            batch_size = self.X_train_tensor.size(1)

            for epoch in range(self.num_epochs):
                model.train()
                optimizer.zero_grad()
                output = model(self.X_train_tensor)
                loss = criterion(output.squeeze(), self.y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 5 == 0:
                    model.eval()
                    with torch.no_grad():
                        test_output = model(self.X_test_tensor)
                        test_loss = criterion(test_output.squeeze(), self.y_test_tensor)
                    print(f'Epoch [{epoch + 1}/{self.num_epochs}], 訓練集 Loss: {loss.item():.4f}, 測試集 Loss: {test_loss.item():.4f}')

            torch.save(model.state_dict(), self.module_path)  # 儲存模型權重

        return model

    def __step3_predict(self, model):
        # 預測並反歸一化
        model.eval()
        with torch.no_grad():
            train_predict = model(self.X_train_tensor).cpu().numpy()
            test_predict = model(self.X_test_tensor).cpu().numpy()

        # 反歸一化
        train_predict = self.scaler.inverse_transform(train_predict)
        y_train_actual = self.scaler.inverse_transform(self.y_train_tensor.cpu().numpy().reshape(-1, 1))
        test_predict = self.scaler.inverse_transform(test_predict)
        y_test_actual = self.scaler.inverse_transform(self.y_test_tensor.cpu().numpy().reshape(-1, 1))

        # 構建完整的時間序列
        predicted = np.concatenate((train_predict, test_predict), axis=0)
        actual = self.scaler.inverse_transform(self.dataset[self.window_width:])

        time = self.dataFrame['Date'].values[-len(actual):]

        train_score = mean_squared_error(y_train_actual, train_predict)
        test_score = mean_squared_error(y_test_actual, test_predict)
        print(f'訓練集 MSE: {train_score:.2f}')
        print(f'測試集 MSE: {test_score:.2f}')

        # 顯示圖表 for console
        if self.showFig:
            # fig.show()
            ans = input("Do you want to show the plot? (y/n)")
            if ans.lower() == "y":
                self.__show_plotly(time, actual, predicted)

        # 創建圖表 for streamlit
        fig = go.Figure()
        # 添加實際價格的線
        fig.add_trace(go.Scatter(x=time, y=actual.flatten(), mode='lines', name='Actual Price', line=dict(color='blue'))    )
        # 添加預測價格的線
        fig.add_trace(go.Scatter(x=time, y=predicted.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))

        # 更新圖表的標題和軸標籤
        fig.update_layout(
            title=f"{self.stock_name} {self.column}",
            xaxis_title="Date",
            yaxis_title=f"{self.column} (USD)",
            xaxis_rangeslider_visible=True
        )

        # 設置 x 軸的刻度顯示
        fig.update_xaxes(dtick='D10', tickformat='%Y-%m-%d', tickangle=45)  # 每n天顯示一次，並旋轉標籤

        # 顯示圖表 for console
        if self.showFig:
            # fig.show()
            ans = input("Do you want to show the fig? (y/n)")
            if ans.lower() == "y":
                fig.show()

        return fig
    
    def __show_plotly(self, time, actual, predicted):
        plt.figure(figsize=(12,6))
        plt.plot(time, actual.flatten(), label='Actual Price', color='blue')
        plt.plot(time, predicted.flatten(), label='Predicted Price', color='red')
        plt.title(f'{self.stock_name} {self.column} Prediction - Transformer Model')
        plt.xlabel('Time')
        plt.ylabel(f'{self.column} (USD)')
        plt.xticks(ticks=range(0, len(time), self.use_data//100), labels=time[::self.use_data//100], rotation=45)  # 每隔5個顯示一個時間標籤
        plt.legend()
        plt.show()


    class _PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(TestTransformer._PositionalEncoding, self).__init__()
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
            super(TestTransformer._TransformerTimeSeries, self).__init__()
            self.model_type = 'Transformer'
            self.input_linear = nn.Linear(feature_size, hidden_dim)  # 新增的線性層
            self.pos_encoder = TestTransformer._PositionalEncoding(d_model=hidden_dim)
            encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim, dropout=dropout)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
            self.decoder = nn.Linear(hidden_dim, 1)
            self.hidden_dim = hidden_dim

        def forward(self, src):
            # src 形狀: (seq_length, batch_size, feature_size)
            src = self.input_linear(src)  # 將輸入映射到 hidden_dim 維度
            src = self.pos_encoder(src)
            output = self.transformer_encoder(src)
            # 取最後一個時間步的輸出
            output = self.decoder(output[-1, :, :])
            return output


if __name__ == '__main__':
    test = TestTransformer('Close')
    test.step_control()
