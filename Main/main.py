import os
from API import StockManager

debug = True
__version__ = "1.2.3"

# This is the main point for testing the code without using streamlit

def main():
    # Get the absolute path and create a StockManager to manage downloaded historical data
    manager = StockManager(abs_path=os.path.dirname(os.path.abspath(__file__)))

    while True:
        while True:
            msg = "Select a function or stock from the list.\n"
            options = {
                "A": "Add New Stock",
                "U": "Update all History",
                "D": "Delete Stock",
            }
            for key, value in options.items():
                msg += f"{key}: {value}\n"

            msg += "Downloaded data:\n"
            if manager.get_stock_list():
                for index, stock in enumerate(manager.msg):
                    msg += f"{index}. {stock}\n"
            else:
                msg += f"{manager.msg}\n"

            ans = input(msg)

            try:
                ans = int(ans)
                index = ans if 0 <= ans < len(manager.stock_class_list) else print("Input error.")
                if index is not None:
                    break

            except ValueError:
                if ans.upper() == "A":
                    stock_name = input("Please enter the stock name (Ex:2330.TW).\n")
                    manager.add_stock(stock_name)
                elif ans.upper() == "U":
                    manager.update_all()
                elif ans.upper() == "D":
                    stock_name = input("Please enter the stock name (Ex:2330.TW).\n")
                    manager.remove_stock(stock_name=stock_name)
                else:
                    print("Input error")

            except Exception as e:
                print(f"Unknown Error: {e}")

        while True:
            options = {
                "I": "Show Company Info",
                "H": "Show History Data",
                "U": "Update History Data",
                "S": "Show Prediction",
                "RT": "Retrain Model",
                "N": "New Model",
                "0": "Return"
            }
            msg = "Select an action.\n"
            for key, value in options.items():
                msg += f"{key}: {value}\n"
            ans = input(msg)

            if ans.upper() == "I":
                # Show Company Info
                info = manager.refresh_company_info(stock_index=index)
                if info:
                    for key in info.keys():
                        print(f"{key:30s} {info[key]}")
            elif ans.upper() == "H":
                # Show History Data
                manager.show_history_data(index)
            elif ans.upper() == "U":
                # Update History Data
                manager.update_history(stock_index=index)
            elif ans.upper() == "S":
                kwargs = {
                    'stock_index': index,
                }
                manager.get_analysis_console(**kwargs)
            elif ans.upper() == "RT":
                kwargs = {
                    'stock_index': index,
                    'retrain_model': True,
                }
                manager.get_analysis_console(**kwargs)
            elif ans.upper() == "N":
                kwargs = {
                    'stock_index': index,
                    'creat_new_model': True,
                }
                manager.get_analysis_console(**kwargs)
            elif ans == "0":
                break

            if debug:
                os.system('pause')
                

if __name__ == "__main__":
    main()
    os.system('pause')
