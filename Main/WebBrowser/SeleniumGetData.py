from selenium import webdriver
from selenium.webdriver.common.by import By
import os
import time

def seleniumGetData(historyDataPath, stockName):
    # Init Setting
    TimePeriod_Xpath = "/html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div/div[2]/div/div/section/div[1]/div[1]/div[1]/div/div/div/span"
    MaxTimeButton_Xpath = "/html/body/div[1]/div/div/div[1]/div/div[3]/div[1]/div/div[2]/div/div/section/div[1]/div[1]/div[1]/div/div/div[2]/div/ul[2]/li[4]/button/span"
    URL = f"https://finance.yahoo.com/quote/{stockName}/history"
    downloadDir = historyDataPath

    # driver setting
    options = webdriver.ChromeOptions()
    # 隱藏視窗
    options.add_argument("headless")

    # 設置下載目錄
    prefs = {"download.default_directory": downloadDir}
    options.add_experimental_option("prefs", prefs)

    # 啟動Chrome
    driver = webdriver.Chrome(options=options)

    try:
        # 打開網頁
        driver.get(URL)
        driver.find_element(By.XPATH, TimePeriod_Xpath).click()
        driver.implicitly_wait(5)
        driver.find_element(By.XPATH, MaxTimeButton_Xpath).click()
        driver.implicitly_wait(5)

        # 準備下載，如果檔案已存在則刪除它
        targetFile = os.path.join(downloadDir, stockName+".csv")
        if os.path.exists(targetFile):
            os.remove(targetFile)

        # 下載
        downloadButton = driver.find_element(By.XPATH, "//span[text()='Download']")
        downloadButton.click()

    except Exception as e:
        raise '連線逾時，股票名稱錯誤或連線錯誤'

    # 等待最多60秒下載時間
    for sec in range(60):
        if os.path.exists(targetFile):
            print('下載完成')
            break
        if sec == 59:
            raise '下載失敗'
        else:
            time.sleep(1)

    # 關閉瀏覽器
    driver.quit()

# 測試用
if __name__ == '__main__':
    absPath = "D:\pycharm\MarketAnalysis\Main"
    seleniumGetData("D:\pycharm\MarketAnalysis\Data", "BTC-USD")
