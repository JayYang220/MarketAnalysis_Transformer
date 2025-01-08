import csv  # 新增匯入csv模組

path1 = 'ref/companyInfo/bitcoin.txt'
path2 = 'ref/companyInfo/stock.txt'

with open(path1, 'r', encoding='utf-8') as file:
    data1 = file.readlines()

with open(path2, 'r', encoding='utf-8') as file:
    data2 = file.readlines()

# 計算交集
intersection = set(data1) & set(data2)


# 計算差集
difference1 = set(data1) - set(data2)
difference2 = set(data2) - set(data1)

print("---------------------------交集---------------------------")
for i in intersection:
    print(i.replace('\n', ''))
print("---------------------------data1的差集---------------------------")
for i in difference1:
    print(i.replace('\n', ''))
print("---------------------------data2的差集---------------------------")
for i in difference2:
    print(i.replace('\n', ''))

company_info_seq = [
    'longName',
    'shortName',

]

intersection_dict = {
    "quoteType": "報價類型",
    "open": "開盤",
    "underlyingSymbol": "標的符號",
    "regularMarketDayHigh": "常規市場日高",
    "shortName": "簡短名稱",
    "exchange": "交易所",
    "fiftyTwoWeekHigh": "52週最高",
    "dayHigh": "日高",
    "firstTradeDateEpochUtc": "首次交易日期（UTC）",
    "twoHundredDayAverage": "200日平均",
    "fiftyDayAverage": "50日平均",
    "timeZoneFullName": "時區全名",
    "regularMarketOpen": "常規市場開盤",
    "averageDailyVolume10Day": "10日平均每日交易量",
    "volume": "交易量",
    "longName": "長名稱",
    "uuid": "UUID",
    "dayLow": "日低",
    "currency": "貨幣",
    "timeZoneShortName": "時區簡稱",
    "regularMarketPreviousClose": "常規市場前收盤",
    "marketCap": "市值",
    "symbol": "符號",
    "averageVolume10days": "10日平均交易量",
    "priceHint": "價格提示",
    "regularMarketDayLow": "常規市場日低",
    "previousClose": "前收盤",
    "averageVolume": "平均交易量",
    "fiftyTwoWeekLow": "52週最低",
    "regularMarketVolume": "常規市場交易量",
    "maxAge": "最大年齡"
}

# 儲存字典為CSV檔案，使用UTF-8 with BOM編碼
with open('intersection_dict.csv', 'w', encoding='utf-8-sig', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Key', 'Value'])  # 寫入標題
    for key, value in intersection_dict.items():
        writer.writerow([key, value])  # 寫入每一對鍵值