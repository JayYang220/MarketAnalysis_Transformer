from common.log import init_logger, log_stream
from common.const import __version__


class init_streamlit:
    manager = None
    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(init_streamlit, cls).__new__(cls)
            from API import StockManager
            cls.instance.manager = StockManager()
            init_logger(logger_name='streamlit')
            log_stream('streamlit', 'info', f'Streamlit mode {__version__} started.')
        return cls.instance
