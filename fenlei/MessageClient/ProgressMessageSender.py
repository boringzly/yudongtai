from kafka import KafkaProducer
import json
import time
import uuid
from copy import deepcopy

class ProgressMessageSender():

    def __init__(self, bootstrap_servers='', topic='', taskId=None):
        try:
            self.producer = KafkaProducer(bootstrap_servers=bootstrap_servers)
        except Exception as exc:
            self.producer = None
            print(f'failed to create sender: {exc!r}')
            return
        self.topic = topic
        if taskId is None:
            taskId = str(uuid.uuid4())
        self.taskId = taskId
        self.msg_dict_default = {
            'messageType': 'progress',
            'sendTime': '0000-00-00 00:00:00',
            'taskId': self.taskId,
        }
        self.titleId = str(uuid.uuid4())
        self.fixed_msg_dict = {
            'version': '3',
            'title': 'unknown',
            'titleId': self.titleId,
            'source': 'default',
            'rank': 0,
        }

    def _build_msg_dict(self, msg_dict):
        _msg_dict = deepcopy(self.msg_dict_default)
        _message_key = []
        _message_content = {}
        for k, v in msg_dict.items():
            _message_key.append(k)
            _message_content[k] = v
        _msg_dict['messageKey'] = _message_key
        _msg_dict['messageContent'] = _message_content
        _msg_dict['sendTime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        return _msg_dict

    def _check_basic_message(self, message_dict):
        if 'progress' not in message_dict:
            message_dict['progress'] = 0
        if 'runningStatus' not in message_dict:
            message_dict['runningStatus'] = 'unknown'
        if 'runningInfo' not in message_dict:
            message_dict['runningInfo'] = 'null'
        return message_dict

    def _append_fixed_message(self, message_dict):
        for k, v in self.fixed_msg_dict.items():
            message_dict[k] = v
        return message_dict

    def is_none(self):
        return self.producer is None

    def get_task_id(self):
        if self.producer is not None:
            return self.taskId

    def set_title(self, title=None, titleId=None):
        if title is not None:
            self.fixed_msg_dict['title'] = title
        if titleId is not None:
            self.fixed_msg_dict['titleId'] = titleId

    def set_source(self, source=None, rank=None):
        if source is not None:
            self.fixed_msg_dict['source'] = source
        if rank is not None:
            self.fixed_msg_dict['rank'] = rank

    def send(self, message_dict):
        if self.producer is None:
            return False
        message_dict = deepcopy(message_dict)
        message_dict = self._check_basic_message(message_dict)
        message_dict = self._append_fixed_message(message_dict)
        message_dict = self._build_msg_dict(message_dict)
        msg = json.dumps(message_dict).encode('utf-8')
        try:
            self.producer.send(self.topic, msg)
            self.producer.flush()
            return True
        except Exception as exc:
            print(f'failed to send message: {exc!r}')
            return False

    def close(self):
        if self.producer is None:
            return
        try:
            self.producer.flush()
            self.producer.close(timeout=0)
        except Exception as exc:
            print(f'failed to close sender: {exc!r}')
        finally:
            self.producer = None

    def calc_progress_value(self, index, total, min_value=0, max_value=100):
        return int(index / total * (max_value - min_value) + min_value)
