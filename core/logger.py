# import logging

# _logger = logging.getLogger(__name__)


# def setup_logging():
#     _logger = logging.getLogger(__name__)
#     _logger.setLevel(logging.DEBUG)
#     handler_m = logging.StreamHandler()
#     formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
#     handler_m.setFormatter(formatter_m)
#     _logger.addHandler(handler_m)
    

import logging

_logger = logging.getLogger(__name__)

def setup_logging():
    _logger = logging.getLogger(__name__)
    _logger.setLevel(logging.DEBUG)
    handler_m = logging.StreamHandler()  # Создаем обработчик для вывода в поток
    formatter_m = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
    handler_m.setFormatter(formatter_m)
    _logger.addHandler(handler_m)