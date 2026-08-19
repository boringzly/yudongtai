import logging
import os


class Logger(object):
    # global formatter for the logger depends on env and the separator is '\t'
    if os.environ.get('CLCORE') is not None:
        aiTaskId = os.environ["AI_TASK_ID"]
        formatter = logging.Formatter(f'CLCore\t%(asctime)s\t%(levelname)-9s\t{aiTaskId}\t%(message)s', datefmt='%m/%d/%Y-%I:%M:%S-%p')
    else:
        formatter = logging.Formatter('%(asctime)s\t%(levelname)-9s\t%(message)s', datefmt='%m/%d/%Y-%I:%M:%S-%p')

    # Logger level type: copy from inner logging module
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    NOTSET = 0

    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        # default
        self._console_log_level = logging.DEBUG

        # creating console handler
        self._ch = None

        self._is_ch_added = False

        # creating file handler
        self._fh = None
        self._file_path = None
        self._fh_log_level = logging.DEBUG
        self._is_fh_added = False

        # map the logging hook to the inner logging  class's method
        self._debug = self._logger.debug
        self._info = self._logger.info
        self._error = self._logger.error
        self._warning = self._logger.warning
        self._warn = self._logger.warn
        self._critical = self._logger.critical
        self._fatal = self._logger.fatal

        # add stage field and process field
        self._stage = None
        self._process = None

        # for appending for the stage and

    def _clearStage(self):
        self._stage = None

    def _getStage(self):
        return self._stage

    def _clearProcess(self):
        self._process = None

    def _getProcess(self):
        return self._process

    def _clearStageAndProcess(self):
        self._clearStage()
        self._clearProcess()

    def setStageAndProcess(self, stage, process):
        self._stage = stage
        self._process = process
        return self

    # wrappers to add fields for msg, maybe python has macros to optimized the code
    def debug(self, msg, *args, **kwargs):
        if self._getStage() is not None and self._getProcess() is not None:
            msg = str(self._stage) + ':' + str(self._process) + '\t' + 'data:' + str(msg)
        self._clearStageAndProcess()
        self._debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self._getStage() is not None and self._getProcess() is not None:
            msg = str(self._stage) + ':' + str(self._process) + '\t' + 'data:' + str(msg)
        self._clearStageAndProcess()
        self._info(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self._getStage() is not None and self._getProcess() is not None:
            msg = str(self._stage) + ':' + str(self._process) + '\t' + 'data:' + str(msg)
        self._clearStageAndProcess()
        self._error(msg, *args, **kwargs)

    def warn(self, msg, *args, **kwargs):
        if self._getStage() is not None and self._getProcess() is not None:
            msg = str(self._stage) + ':' + str(self._process) + '\t' + 'data:' + str(msg)
        self._clearStageAndProcess()
        self._warn(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self._getStage() is not None and self._getProcess() is not None:
            msg = str(self._stage) + ':' + str(self._process) + '\t' + 'data:' + str(msg)
        self._clearStageAndProcess()
        self._warning(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if self._getStage() is not None and self._getProcess() is not None:
            msg = str(self._stage) + ':' + str(self._process) + '\t' + 'data:' + str(msg)
        self._clearStageAndProcess()
        self._critical(msg, *args, **kwargs)

    def fatal(self, msg, *args, **kwargs):
        if self._getStage() is not None and self._getProcess() is not None:
            msg = str(self._stage) + ':' + str(self._process) + '\t' + 'data:' + str(msg)
        self._clearStageAndProcess()
        self._fatal(msg, *args, **kwargs)

    def setBasicLogLevel(self, level=logging.INFO):
        self._logger.setLevel(level)
        return self

    def setConsoleLogLevel(self, level=logging.INFO):
        if self._is_ch_added:
            # we assume the console handler is added, we remove it add recreated it
            self._logger.removeHandler(self._ch)
            # new console handler, stderr is used
            self._console_log_level = level
            self._ch = logging.StreamHandler()
            self._ch.setLevel(self._console_log_level)
            self._ch.setFormatter(self.formatter)
            self._logger.addHandler(self._ch)
        else:
            self._ch = logging.StreamHandler()
            self._console_log_level = level
            self._ch = logging.StreamHandler()
            self._ch.setLevel(self._console_log_level)
            self._ch.setFormatter(self.formatter)
            self._logger.addHandler(self._ch)
            self._is_ch_added = True
        return self

    def setFilePathAndLogLevel(self, path, level=logging.INFO):
        if self._is_fh_added:
            self._logger.removeHandler(self._fh)
            self._file_path = path
            self._fh = logging.FileHandler(path)
            self._fh_log_level = level
            self._fh.setLevel(self._fh_log_level)
            self._fh.setFormatter(self.formatter)
            self._logger.addHandler(self._fh)
        else:
            self._file_path = path
            self._fh = logging.FileHandler(path)
            self._fh_log_level = level
            self._fh.setLevel(self._fh_log_level)
            self._fh.setFormatter(self.formatter)
            self._logger.addHandler(self._fh)
            self._is_fh_added = True
        return self

    def hasFileLog(self):
        return self._is_fh_added

    def hasConsoleLog(self):
        return self._is_ch_added

    def removeFileLog(self):
        if self._is_fh_added:
            self._logger.removeHandler(self._fh)
        self._is_fh_added = False
        return self

    def removeConsoleLog(self):
        if self._is_ch_added:
            self._logger.removeHandler(self._ch)
        self._is_ch_added = False
        return self


# global logger signature for exporting
logger = Logger().setBasicLogLevel(Logger.DEBUG).setConsoleLogLevel(Logger.INFO)

if __name__ == '__main__':
    logger.debug("this line will not output, because the default log level is warning")
    logger.warning("warning info")
    logger.setBasicLogLevel(Logger.DEBUG).setConsoleLogLevel(Logger.DEBUG)
    logger.debug("debug info")

    # append file log to ./a.log the log level is Logger.INFO
    logger.setFilePathAndLogLevel("./a.log", Logger.INFO)
    logger.info("this will output to a.log and console")

    logger.removeFileLog()
    logger.info("this will only log to console, because we remove the file hander")
    logger.removeConsoleLog()
    logger.warning("this will use the default console log ,log level is warning!!! and remove the file formatter!!")

    # this is user case: console log and file logg all in one
    logger = logger.setBasicLogLevel(Logger.INFO).setConsoleLogLevel(Logger.INFO). \
        setFilePathAndLogLevel("./a.log", logging.INFO)

    # set stage and process
    logger.setStageAndProcess("START_LOGGER", "5%").debug("the data field")

    logger.warning('warning info')
    logger.setConsoleLogLevel(logging.DEBUG)
    logger.debug("debug info")

    # the wrapper log: auto add stage adn process field and modifed the data field, the sep is "TAB"
    logger.setStageAndProcess("FINISH_JOB", "100%").info("the data field")

    # the common log, just add CLCore(if we add env variable) timestamp, log level , info
    logger.warning("warning info")
    logger.fatal("fatal info")

    logger.info("hello world")
    logger.setStageAndProcess("FINISH_JOB", "100%").info("the data field")
    logger.info("aaa")
