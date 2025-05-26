import logging
import time

def print_log(title: str, content: str, level: int = 2):
    logging.info(time.asctime())
    if level == 0:
        logging.info(f'========================= {title} =========================')
    elif level == 1:
        logging.info(f'---------- {title} ----------')
    else:
        logging.info(f'>>> {title}')

    logging.info(content)


def init_log(log_file: str, level: str = 'info', terminal: bool = False, clear: bool = False, with_prefix: bool = False):
    level = logging.DEBUG if level.lower() == 'debug' else logging.INFO

    if clear:
        with open(log_file, 'w'):
            pass

    logger = logging.getLogger()
    # remove all handlers
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    logger.setLevel(level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    if with_prefix:
        formatter = logging.Formatter(
            fmt='''[%(asctime)s - %(levelname)s]: %(message)s''',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    if terminal:
        logger.addHandler(console_handler)
