import time
from tqdm import tqdm
import sys

class Logger_v2():

    def __init__(self):
        # log file path. setting None for log to screen only
        self._file_path = None
        # debug flag. setting False for not output debug content.
        self._debug_flag = False
        # brand info. show your framework's name. default is CLCore, recommand changing to CLDet, CLSeg, CLCD, etc.
        self._brand = 'CLCore'

    # inner method. change debug flag.
    def _set_debug(self, flag):
        self._debug_flag = flag

    # inner method. core print function
    def _log(self, level, text, print_type='print'):
        # process text
        _time_str = time.strftime('%y%m%d-%H%M%S', time.localtime())
        _text_str = f'[{self._brand}] [{level}] [{_time_str}] {text}'
        # print or tqdm
        if print_type == 'print':
            print(_text_str)
        elif print_type == 'tqdm':
            tqdm.write(_text_str)
        # output to log file
        if self._file_path is not None:
            try:
                with open(self._file_path, 'a') as f:
                    f.write(f'{_text_str}\n')
            except:
                _time_str = time.strftime('%y%m%d-%H%M%S', time.localtime())
                print(f'[{self._brand}] [WARNING] [{_time_str}] Can NOT open {self._file_path}')

    # change brand name
    def set_brand(self, brand):
        self._brand = brand

    # setting log file path. setting initial to True for try to create the log file, in case the path is not operated.
    def set_file_path(self, path=None, initial=False):
        if path is not None:
            self._file_path = path
            if initial:
                try:
                    with open(self._file_path, 'w') as f:
                        f.write('')
                except:
                    _time_str = time.strftime('%y%m%d-%H%M%S', time.localtime())
                    print(f'[{self._brand}] [WARNING] [{_time_str}] Can NOT open {self._file_path}')  

    # debug, info, warning, error for print
    def debug(self, text):
        if self._debug_flag:
            self._log('DEBUG', text, print_type='print')

    def info(self, text):
        self._log('INFO', text, print_type='print')

    def warning(self, text):
        self._log('WARNING', text, print_type='print')

    def error(self, text):
        self._log('ERROR', text, print_type='print')
        self._log('ERROR', 'Exit program!', print_type='print')
        sys.exit(-1)

    # debug, info, warning, error for tqdm
    def tdebug(self, text):
        if self._debug_flag:
            self._log('DEBUG', text, print_type='tqdm')

    def tinfo(self, text):
        self._log('INFO', text, print_type='tqdm')

    def twarning(self, text):
        self._log('WARNING', text, print_type='tqdm')

    def terror(self, text):
        self._log('ERROR', text, print_type='tqdm')
        self._log('ERROR', 'Exit program!', print_type='tqdm')
        sys.exit(-1)


logger_v2 = Logger_v2()


if __name__ == '__main__':
    logger.set_brand('CLTest')
    logger.set_file_path('/tmp/test.log')
    logger.debug('debug content')
    logger.info('info content')
    logger.warning('warning content')
    #logger.error('error content')
    logger._set_debug(True)
    logger.debug('debug content')
    logger.info('info content')
    logger.warning('warning content')
    logger.error('error content')