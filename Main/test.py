# Main/test.py
class testCall:
    args = []  # 將 args 設為類屬性

    @classmethod
    def restore(cls, *args):
        cls.args = args  # 儲存傳入的參數

    @classmethod
    def write(cls):
        print(*cls.args)  # 使用 print 輸出參數

# 直接使用類別調用方法
testCall.restore(1, 2, 3)
testCall.write()