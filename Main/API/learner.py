import pandas as pd
import numpy as np
import math

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

# https://ithelp.ithome.com.tw/articles/10361025
# https://medium.com/ching-i/transformer-attention-is-all-you-need-c7967f38af14
# https://medium.com/ching-i/pytorch-%E5%9F%BA%E6%9C%AC%E4%BB%8B%E7%B4%B9%E8%88%87%E6%95%99%E5%AD%B8-ac0e1ebfd7ec

class ModelControl:
    def __init__(self, **kwargs):
        self.stock_name = kwargs['stock_name']
        self.column = kwargs['column']
        self.output_func = kwargs.get('output_func', print)

        self.use_data = 1000 # 使用最後n筆資料量
        self.dataFrame = pd.read_csv(kwargs['history_data_path'])[-self.use_data:]
        self.src_model_path = kwargs.get('src_model_path', None)
        self.dst_model_path = kwargs.get('dst_model_path', None)
        self.train_size = 0.8
        self.test_size = 1 - self.train_size

        self.test_length = 0
        
        self.window_width = kwargs.get('window_width', 60) #60
        self.num_epochs = kwargs.get('num_epochs', 100) #100

        self.num_layers = kwargs.get('num_layers', 2) # 2
        self.nhead = kwargs.get('nhead', 4) # 4 會影響模型預測
        self.hidden_dim = kwargs.get('hidden_dim', 128) #128
        self.dropout = kwargs.get('dropout', 0.1) #0.1

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = None
        self.model = None
        self.X_train_tensor, self.y_train_tensor = None, None
        self.X_test_tensor, self.y_test_tensor = None, None

        self.create_new_model = kwargs['create_new_model']
        self.retrain_model = kwargs['retrain_model']
        self.showFig = kwargs.get('show_fig', None)
        self.showGoFig = kwargs.get('show_go_fig', None)
        self.display_epochs = kwargs.get('display_epochs', 10)

        self.fig = None
        self.plot_time = None
        self.plot_actual = None
        self.plot_predicted = None
        self.train_MSE = None
        self.test_MSE = None

    def start(self):
        self.output_func("Preprocessing data...")
        data = self.dataFrame[[self.column]].copy()
        # data[self.column] = data[self.column].pct_change() # 計算百分比變化
        # data[self.column] = data[self.column].fillna(1)  # 將NaN值轉換為1
        data[self.column] = self.scaler.fit_transform(data[[self.column]])
        self.X_train_tensor, self.X_test_tensor, self.y_train_tensor, self.y_test_tensor = self.__step1_preprocess_data(data)

        # X_train 用於訓練
        # X_test 用於訓練
        # y_train 是用於比對預測偏差
        # y_test 是用於比對預測偏差
        
        self.output_func("Building/Loading model...")
        self.model = self.__step2_build_model()
        if self.create_new_model or self.retrain_model:
            return

        self.output_func("Predicting...")
        self.__step3_predict(self.model)
        self.output_func("Train MSE:", self.train_MSE)
        self.output_func("Test MSE:", self.test_MSE)
        self.fig = self.__step4_plot()
    
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

        # x 是所有窗口的資料
        # y 是實際的數值

        # 劃分比例 訓練及測試
        train_size = int(len(x) * self.train_size)
        test_size = len(x) - train_size

        X_train = x[:train_size]
        X_test = x[train_size:]

        y_train = y[:train_size]
        y_test = y[train_size:]

        self.test_length = len(X_test)

        # 將數據轉換為 PyTorch 張量
        X_train_tensor = torch.from_numpy(X_train).float()
        y_train_tensor = torch.from_numpy(y_train).float()
        X_test_tensor = torch.from_numpy(X_test).float()
        y_test_tensor = torch.from_numpy(y_test).float()

        # 調整輸入形狀為 (seq_length, batch_size, feature_size)
        X_train_tensor = X_train_tensor.unsqueeze(-1).permute(1, 0, 2)
        X_test_tensor = X_test_tensor.unsqueeze(-1).permute(1, 0, 2)

        return X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor
    
    def __step2_build_model(self):
        feature_size = 1
        num_layers = self.num_layers
        nhead = self.nhead
        hidden_dim = self.hidden_dim
        dropout = self.dropout
        device = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        print(f"You are using: {device}")
        
        model = ModelControl._TransformerTimeSeries(feature_size=feature_size, num_layers=num_layers, nhead=nhead, hidden_dim=hidden_dim, dropout=dropout).to(device)

        if not self.create_new_model:
            model.load_state_dict(torch.load(self.src_model_path))

        if self.create_new_model or self.retrain_model:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # 轉換數據為 PyTorch 的 Tensor 並移動到設備
            self.X_train_tensor = self.X_train_tensor.to(device)
            self.X_test_tensor = self.X_test_tensor.to(device)

            self.y_train_tensor = self.y_train_tensor.to(device)
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
                
                if (epoch + 1) % self.display_epochs == 0 or epoch == self.num_epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        test_output = model(self.X_test_tensor)
                        test_loss = criterion(test_output.squeeze(), self.y_test_tensor)
                    self.output_func(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')

            torch.save(model.state_dict(), self.dst_model_path)  # 儲存模型權重

        return model

    def __step3_predict(self, model):
        # 預測並反歸一化
        model.eval()
        with torch.no_grad():
            train_predict = model(self.X_train_tensor).cpu().numpy()
            print(self.X_test_tensor.shape)
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

        self.plot_time = time
        self.plot_actual = actual
        self.plot_predicted = predicted

        train_score = mean_squared_error(y_train_actual, train_predict)
        test_score = mean_squared_error(y_test_actual, test_predict)

        self.train_MSE = train_score
        self.test_MSE = test_score

    def __step4_plot(self):
        # 顯示圖表 for console
        if self.showFig is None:
            # fig.show()
            ans = input("Do you want to show the plot? (y/n)")
            if ans.lower() == "y":
                self.__show_plotly(self.plot_time, self.plot_actual, self.plot_predicted)
        elif self.showFig is True:
            self.__show_plotly(self.plot_time, self.plot_actual, self.plot_predicted)

        fig = self.__get_go_fig(self.plot_time, self.plot_actual, self.plot_predicted)
        if self.showGoFig is None:
            ans = input("Do you want to show the fig? (y/n)")
            if ans.lower() == "y":
                fig.show()
        elif self.showGoFig is True:
            fig.show()
        
        return fig
    
    def __show_plotly(self, time, actual, predicted):
        plt.figure(figsize=(12,6))
        plt.plot(time[-self.test_length:], actual.flatten()[-self.test_length:], label='Actual Price', color='blue')
        plt.plot(time[-self.test_length:], predicted.flatten()[-self.test_length:], label='Predicted Price', color='red')
        plt.title(f'{self.stock_name} {self.column} Prediction - Transformer Model')
        plt.xlabel('Time')
        plt.ylabel(f'{self.column} (USD)')
        plt.xticks(ticks=range(0, len(time[-self.test_length:]), self.use_data//100), labels=time[-self.test_length::self.use_data//100], rotation=45)  # 每隔5個顯示一個時間標籤
        plt.legend()
        plt.show()

    def __get_go_fig(self, time, actual, predicted):
        # 創建圖表 for streamlit
        fig = go.Figure()
        # 添加實際價格的線
        fig.add_trace(go.Scatter(x=time, y=actual.flatten(), mode='lines', name='Actual Price', line=dict(color='blue'))    )
        # 添加預測價格的線
        fig.add_trace(go.Scatter(x=time, y=predicted.flatten(), mode='lines', name='Predicted Price', line=dict(color='red')))

        # 更新圖表的標題和軸標籤
        fig.update_layout(
            title=f"{self.stock_name} {self.column} - Train MSE: {self.train_MSE:.4f}, Test MSE: {self.test_MSE:.4f}",
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
            while predict_days > -1:
                # src 形狀: (seq_length, batch_size, feature_size)
                src2 = self.input_linear(src)  # 將輸入映射到 hidden_dim 維度
                src2 = self.pos_encoder(src2)  # Positional Encoding 就是位置編碼，目的是為了讓模型考慮位置之間的順序
                output_org = self.transformer_encoder(src2)
                # 取最後一個時間步的輸出
                output = self.decoder(output_org[-1, :, :])
                src = torch.cat((src, output.unsqueeze(0)), dim=0)
                src = src[-60:]

                predict_days -= 1

            return output
        
            # 預測��來5日
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
