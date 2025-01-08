import os
import logging
from datetime import datetime

MODE = ['debug', 'info', 'warning', 'error', 'critical']

def init_logger(logger_name:str):
    """
    初始化1個日誌記錄器
    :param logger_name: 日誌記錄器名稱
    """
    log_mode = os.getenv('LOG_MODE')
    if log_mode is None:
        log_mode = 'info'
    elif log_mode not in MODE:
        raise ValueError(f"Invalid mode: {log_mode}. Must be one of: {MODE}")
    
    log_path = os.getenv('ROOT_PATH') + r'\\log'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # 设置 logger
    logger = logging.getLogger(logger_name)
    if log_mode == 'debug':
        logger.setLevel(logging.DEBUG)
    elif log_mode == 'info':
        logger.setLevel(logging.INFO)
    elif log_mode == 'warning':
        logger.setLevel(logging.WARNING)
    elif log_mode == 'error':
        logger.setLevel(logging.ERROR)
    elif log_mode == 'critical':
        logger.setLevel(logging.CRITICAL)

    # 创建 file handlers
    now = datetime.now()
    hundredth_of_second = now.microsecond // 10000 
    file_date = now.strftime("%Y%m%d%H%M%S") + f"{hundredth_of_second:02d}"

    handler = logging.FileHandler(log_path + rf'\\{file_date}_{logger_name}.log', mode='w', encoding='utf-8')

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # 添加 handler 到 logger
    logger.addHandler(handler)

    # 添加 handler 到 console
    logger.addHandler(logging.StreamHandler())

def log_stream(logger_name:str, level:str, msg, *args, **kwargs):
    """
    输出日誌到控制台和文件
    :param logger_name: 日誌記錄器名稱
    :param level: 日誌級別
    :param msg: 日誌消息
    :param args: [logger參數](https://docs.python.org/3/library/logging.html#logging.Logger)
    :param kwargs: [logger參數](https://docs.python.org/3/library/logging.html#logging.Logger)
    """

    if level not in MODE:
        raise ValueError(f"Invalid level: {level}. Must be one of: {MODE}")
    if logger_name not in logging.root.manager.loggerDict:
        raise ValueError(f"Invalid logger_name: {logger_name}.")
    else:
        logger = logging.getLogger(logger_name)

    if isinstance(msg, str):
        msg_list = [msg]
    elif isinstance(msg, list):
        msg_list = msg
    else:
        raise ValueError(f"Invalid msg: {msg}. Must be a string or a list of strings.")
    
    for m in msg_list:
        if level == 'debug':
            logger.debug(m, *args, **kwargs)
        elif level == 'info':
            logger.info(m, *args, **kwargs)
        elif level == 'warning':
            logger.warning(m, *args, **kwargs)
        elif level == 'error':
            logger.error(m, *args, **kwargs)
        elif level == 'critical':
            logger.critical(m, *args, **kwargs)