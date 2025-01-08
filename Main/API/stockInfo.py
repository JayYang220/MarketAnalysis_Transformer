# Guide https://pypi.org/project/yfinance/
# Guide https://algotrading101.com/learn/yahoo-finance-api-guide/
import yfinance as yf
import os
import pandas as pd
from .learner import ModelControl
from common import log_stream
from typing import Callable
import datetime
import pytz
import traceback

is_debug = True
ROOT_PATH = os.getenv('ROOT_PATH')
MODE = os.getenv('MODE')

# ref = refresh

class StockManager:
    def __init__(self):
        self.history_data_folder_path = os.path.join(ROOT_PATH, "data")
        self.model_folder_path = os.path.join(ROOT_PATH, "model")
        self.__init_dir()

        # 讀取庫存的CSV和model名稱 集中成list管理
        self.stock_name_list = self.__load_stock_list()
        self.model_name_list = self.__load_model_list()

        # 建立stock class 集中成list管理
        self.stock_class_list: list[Stock] = self.__init_create_stock_class()

        self.msg = []
        self.is_action_successful = False

    def __new__(cls):
        """singleton"""
        if not hasattr(cls, 'instance'):
            cls.instance = super(StockManager, cls).__new__(cls)
            cls.instance.__init__()
        return cls.instance

    def __init_dir(self):
        """檢查並建立基本資料夾"""
        if not os.path.exists(self.history_data_folder_path):
            os.mkdir(self.history_data_folder_path)
        if not os.path.exists(self.model_folder_path):
            os.mkdir(self.model_folder_path)

    def __load_stock_list(self) -> list[str]:
        """
        讀取庫存的CSV名稱 集中成list管理
        此函式僅在初期建立obj時使用，其餘增刪修都由其他函式負責
        """
        file_list = os.listdir(self.history_data_folder_path)
        stock_list = []
        for file in file_list:
            if file.endswith('.csv'):
                stock_list.append(os.path.splitext(file)[0])
        return stock_list

    def __load_model_list(self) -> list[str]:
        """
        讀取庫存的模型名稱 集中成list管理
        此函式僅在初期建立obj時使用，其餘增刪修都由其他函式負責
        """
        file_list = os.listdir(self.model_folder_path)
        model_list = []
        for file in file_list:
            if file.endswith('.pth'):
                model_list.append(os.path.splitext(file)[0])
        return model_list
    
    def rename_model(self, model_name: str, new_model_name: str, output_func: Callable[[str], None]):
        """重命名模型"""
        self.is_action_successful = True

        try:
            os.rename(os.path.join(self.model_folder_path, model_name + ".pth"), os.path.join(self.model_folder_path, new_model_name + ".pth"))
            os.rename(os.path.join(self.model_folder_path, model_name + ".json"), os.path.join(self.model_folder_path, new_model_name + ".json"))
            self.model_name_list[self.model_name_list.index(model_name)] = new_model_name
            output_func(f"{model_name}: Rename completed.")
            log_stream(MODE, 'info', f"{model_name}: Rename completed.")
            return True
            
        except Exception as e:
            output_func(f"{model_name}: {e}")
            log_stream(MODE, 'error', f"{model_name}: {e}")
            log_stream(MODE, 'debug', traceback.format_exc())
            self.is_action_successful = False
            return False
    
    def remove_model(self, model_name: str, output_func: Callable[[str], None]):
        """移除模型"""
        self.is_action_successful = True
        try:
            os.remove(os.path.join(self.model_folder_path, model_name + ".pth"))
            if os.path.exists(os.path.join(self.model_folder_path, model_name + ".json")):
                os.remove(os.path.join(self.model_folder_path, model_name + ".json"))
            self.model_name_list.remove(model_name)
            output_func(f"{model_name}: Remove completed.")
            log_stream(MODE, 'info', f"{model_name}: Remove completed.")
            return True
        except Exception as e:
            output_func(f"{model_name}: {e}")
            log_stream(MODE, 'error', e)
            log_stream(MODE, 'debug', traceback.format_exc())
            self.is_action_successful = False
            return False

    def __init_create_stock_class(self):
        """建立stock class 集中成list管理"""
        stock_class_list = []
        for stock_name in self.stock_name_list:
            stock_class_list.append(Stock(self.history_data_folder_path, self.model_folder_path, stock_name))
        return stock_class_list
    
    def get_stock_index(self, stock_name: str) -> int:
        """
        取得stock_name在stock_name_list的index
        :param stock_name: str
        :return: index if stock_name in stock_name_list else -1
        """
        try:
            return self.stock_name_list.index(stock_name)
        except ValueError:
            return -1
        
    def get_stock_data_len(self, stock_name: str):
        """取得stock_name的data length"""
        return self.stock_class_list[self.get_stock_index(stock_name)].get_data_len()

    def add_stock(self, stock_name: str, output_func: Callable[[str], None]):
        """try create"""
        self.is_action_successful = True
        if stock_name in self.stock_name_list:
            self.is_action_successful = False
            log_stream(MODE, 'info', f"{stock_name}: This stock already exists in the list.")
            output_func(f"{stock_name}: This stock already exists in the list.")
            return

        # try downloading ticker
        try:
            stock = Stock(self.history_data_folder_path, self.model_folder_path, stock_name)

            if stock._download_ticker():
                stock.download_history_data()

                log_stream(MODE, 'info', f"{stock_name}: Addition completed")
                output_func(f"{stock_name}: Addition completed")

                self.stock_class_list.append(stock)
                self.stock_name_list.append(stock_name)
                self.is_action_successful = True
            else:
                output_func(f"{stock_name}: Stock name error. You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/.")
                self.is_action_successful = False

        except Exception as e:
            self.is_action_successful = False
            log_stream(MODE, 'error', e)
            log_stream(MODE, 'debug', traceback.format_exc())
            output_func(f"{stock_name}: {e}")
            return

    def update_history(self, **kwargs):
        self.is_action_successful = True
        # for streamlit
        if "stock_name" in kwargs:
            stock_index = self.get_stock_index(kwargs["stock_name"])
        # for console
        elif "stock_index" in kwargs:
            stock_index = kwargs["stock_index"]
        else:
            self.is_action_successful = False
            raise ValueError("stock_name or stock_index is required")
        
        if "output_func" in kwargs:
            output_func = kwargs["output_func"]
        else:
            self.is_action_successful = False
            raise ValueError("output_func is required")
        
        output_func(self.stock_class_list[stock_index].download_history_data())

    def update_all(self, output_func: Callable[[str], None]):
        """更新所有股票資訊"""
        self.is_action_successful = True
        if self.stock_class_list:
            for stock in self.stock_class_list:
                msg = stock.download_history_data()
                log_stream(MODE, 'info', msg)
                output_func(msg)
        else:
            msg = "There is no data in the system."
            log_stream(MODE, 'info', msg)
            self.is_action_successful = False
            output_func(msg)

    def remove_stock(self, **kwargs):
        self.is_action_successful = True
        try:
            self.msg.clear()
            # for streamlit
            if "stock_name" in kwargs:
                stock_name = kwargs["stock_name"]
                stock_index = self.get_stock_index(kwargs["stock_name"])
            # for console
            elif "stock_index" in kwargs:
                stock_name = self.stock_name_list[kwargs["stock_index"]]
                stock_index = kwargs["stock_index"]
            else:
                self.is_action_successful = False
                raise ValueError("stock_name or stock_index is required")
            
            if "output_func" in kwargs:
                output_func = kwargs["output_func"]
            else:
                self.is_action_successful = False
                raise ValueError("output_func is required")
            
            path = self.stock_class_list[stock_index].history_data_file_path
            os.remove(path)

            self.stock_name_list.remove(stock_name)
            self.stock_class_list.pop(stock_index)
            output_func(f"{stock_name}: Remove completed.")
            log_stream(MODE, 'info', f"{stock_name}: Remove completed.")

            self.is_action_successful = True
        except Exception as e:
            output_func(f"{stock_name}: {e}")
            log_stream(MODE, 'error', e)
            log_stream(MODE, 'debug', traceback.format_exc())
            self.is_action_successful = False

    # for streamlit, console
    def get_stock_list(self):
        """顯示庫存的CSV名稱"""
        self.msg.clear()
        if self.stock_name_list:
            for i in self.stock_name_list:
                self.msg.append(i)
            return True
        else:
            log_stream(MODE, 'info', "No Data in the system.")
            self.msg.append("No Data in the system.")
            return False

    # for console
    def show_history_data(self, stock_index: int):
        """顯示 HistoryData"""
        if self.stock_class_list[stock_index].history_data is None:
            try:
                self.stock_class_list[stock_index].history_data = pd.read_csv(self.stock_class_list[stock_index].history_data_file_path)
            except Exception as e:
                log_stream(MODE, 'error', e)
                log_stream(MODE, 'debug', traceback.format_exc())
                return e
        
    def refresh_company_info(self, **kwargs):
        self.msg.clear()
        # for streamlit
        if "stock_name" in kwargs:
            stock_index = self.get_stock_index(kwargs["stock_name"])
            stock_name = kwargs["stock_name"]
        # for console
        elif "stock_index" in kwargs:
            stock_index = kwargs["stock_index"]
            stock_name = self.stock_name_list[stock_index]
        else:
            raise ValueError("stock_name or stock_index is required")
        
        try:
            if self.stock_class_list[stock_index].download_company_info():
                self.msg.append(self.stock_class_list[stock_index].company_info.copy())

                log_stream(MODE, 'info', f"{stock_name}: {self.msg[-1]}")
                self.is_action_successful = True
                return True
            else:
                self.is_action_successful = False
                return False

        except Exception as e:
            self.msg.append(f"{stock_name}: {e}")
            log_stream(MODE, 'error', e)
            log_stream(MODE, 'debug', traceback.format_exc())
            self.is_action_successful = False
            return False
    
    # for console
    def get_analysis_console(self, **kwargs) -> bool:
        kwargs['show_fig'] = None
        kwargs['show_go_fig'] = None
        kwargs['stock_name'] = self.stock_name_list[kwargs['stock_index']]
        kwargs['src_model_path'] = self.stock_class_list[kwargs['stock_index']].model_file_path
        kwargs['dst_model_path'] = self.stock_class_list[kwargs['stock_index']].model_file_path

        creat_new_model = kwargs.get('creat_new_model', False)
        retrain_model = kwargs.get('retrain_model', False)
        kwargs['create_new_model'] = creat_new_model
        kwargs['retrain_model'] = retrain_model

        if os.path.exists(kwargs['dst_model_path']):
            if creat_new_model:
                ans = input("Model file already exists. Do you want to overwrite it? (y/n)")
                if ans.lower() != "y":
                    return
        elif retrain_model:
                print("Model file not found. Please create a new model first.")
                return
        else:
            kwargs['creat_new_model'] = True
        

        return self.__show_prediction(**kwargs)

    # for streamlit
    def get_analysis(self, **kwargs) -> bool:
        kwargs['show_fig'] = False
        kwargs['show_go_fig'] = False
        kwargs['stock_index'] = self.get_stock_index(kwargs['stock_name'])

        if kwargs['src_model_name'] is not None:
            kwargs['src_model_path'] = os.path.join(self.model_folder_path, kwargs['src_model_name']) + ".pth"
        else:
            kwargs['src_model_path'] = None
        
        if 'dst_model_name' in kwargs:
            kwargs['dst_model_path'] = os.path.join(self.model_folder_path, kwargs['dst_model_name']) + ".pth"

        return self.__show_prediction(**kwargs)
    
    def __show_prediction(self, **kwargs):
        kwargs['history_data_path'] = self.stock_class_list[kwargs['stock_index']].history_data_file_path
        kwargs['column'] = "Close"

        a = ModelControl(**kwargs)
        a.start()

        # 可能會有增減model，更新model_name_list
        self.model_name_list = self.__load_model_list()
        return a._fig

class Stock:
    def __init__(self, history_data_folder_path: str, model_folder_path: str, stock_name: str):
        self.stock_name = stock_name

        # 建立歷史資料檔案路徑(含副檔名)
        self.history_data_file_path = os.path.join(history_data_folder_path, self.stock_name + ".csv")

        # 建立模型檔案路徑(含副檔名)
        self.model_file_path = os.path.join(model_folder_path, self.stock_name + ".pth")

        # 初始化為None，待使用者輸入需求時再抓取
        self.ticker = None
        self.company_info = {}
        self.history_data = None

    def get_data_len(self) -> int:
        """取得歷史資料長度"""
        return len(pd.read_csv(self.history_data_file_path))
    
    def _download_ticker(self) -> bool:
        """下載 ticker"""

        self.ticker = yf.Ticker(self.stock_name)

        # 股票名稱錯誤時，仍會返回一個dict，利用下列特徵確認股票名稱是否正確
        if 'previousClose' not in self.ticker.info:
            log_stream(MODE, 'error', f"Failed to download ticker for {self.stock_name}.")
            return False
        else:
            log_stream(MODE, 'info', f'Downloaded ticker for {self.stock_name}.')
            return True
        
    def __format_company_info(self):
        """格式化company_info"""
        # 保留的key與對應名稱
        keys_dict = {'symbol': 'Symbol', 'shortName': 'Short Name', 'longName': 'Long Name', 'timeZoneFullName': 'Time Zone',
                    'timeZoneShortName': 'Time Zone (UTC)', 'firstTradeDateEpochUtc': 'First Trade Date', 'exchange': 'Exchange',
                    'quoteType': 'Quote Type', 'currency': 'Currency', 'marketCap': 'Market Cap', 'open': 'Open',
                    'previousClose': 'Previous Close', 'dayHigh': 'Day High', 'dayLow': 'Day Low', 'fiftyDayAverage': '50 Days Average',
                    'twoHundredDayAverage': '200 Days Average', 'fiftyTwoWeekHigh': '52 Weeks High', 'fiftyTwoWeekLow': '52 Weeks Low',
                    'volume': 'Volume', 'averageVolume': 'Average Volume', 'averageVolume10days': 'Average Volume 10 Days'}
        
        for key in keys_dict.keys():
            try:
                if key in self.ticker.info:
                    if key == 'firstTradeDateEpochUtc':
                        self.company_info[keys_dict[key]] = datetime.datetime.fromtimestamp(self.ticker.info[key])
                    elif key == "marketCap" or key == "volume" or key == "averageVolume" or key == "averageVolume10days":
                        self.company_info[keys_dict[key]] = "{:,}".format(self.ticker.info[key])
                    elif key == "timeZoneShortName":
                        self.company_info[keys_dict[key]] = self.__get_utc_offset(self.ticker.info[key])
                    else:
                        self.company_info[keys_dict[key]] = self.ticker.info[key]
            except Exception as e:
                log_stream(MODE, 'error', f"{self.stock_name}, format_company_info key error: {key}: {e}")
                log_stream(MODE, 'debug', traceback.format_exc())

    @staticmethod
    def __get_utc_offset(timezone: str) -> str:
        """根據時區名稱返回UTC偏移量"""
        try:
            tz = pytz.timezone(timezone)
            offset = tz.utcoffset(datetime.datetime.now()).total_seconds() / 3600
            return f"{timezone} (UTC{offset:+.0f})"
        except Exception as e:
            log_stream(MODE, 'error', f"{timezone}, get_utc_offset error: {e}")
            log_stream(MODE, 'debug', traceback.format_exc())
            return timezone

    def download_company_info(self) -> bool:
        # 強制更新ticker
        if self._download_ticker():
            self.__format_company_info()
            return True
        else:
            return False
        
    def download_history_data(self, period: str = 'max', interval: str = '1d') -> bool:
        """Download HistoryData"""
        try:
            if self._download_ticker():
                self.history_data = self.ticker.history(period=period, interval=interval)
                self.history_data['Date'] = pd.to_datetime(self.history_data.index).strftime('%Y-%m-%d')
                self.history_data.to_csv(self.history_data_file_path, index=False)
                log_stream(MODE, 'info', f"{self.stock_name}: Update completed.")
                return f"{self.stock_name}: Update completed."
            else:
                log_stream(MODE, 'error', "Stock name error. You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/")
                return f"{self.stock_name}: Stock name error. You can retrieve stock names from Yahoo Finance. https://finance.yahoo.com/."
        except Exception as e:
            return f"{self.stock_name}: {e}"

