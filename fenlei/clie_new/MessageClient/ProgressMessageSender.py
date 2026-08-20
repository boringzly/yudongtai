from kafka import KafkaProducer
import json
import time
import uuid
from copy import deepcopy

class ProgressMessageSender():
    
    def __init__(self, bootstrap_servers='', topic='', taskId=None):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self._next_connect_time = 0
        self._pending_message = None
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
        '''
        @message template
        progress: 0 @int
        runningStatus: running @str
        runningInfo: starting @str
        inferProgress: 0 @int
        inferFilename: somefile.tif @str
        inferGeoInfo: POLYGON(({x1} {y1}, {x2} {y2}, {x3} {y3}, {x4} {y4}, {x1} {y1})) @str - list
        inferObjectGeoInfo: POLYGON(({x1} {y1}, {x2} {y2}, {x3} {y3}, {x4} {y4}, {x1} {y1})) @str - list
        '''

    def _connect(self, force=False):
        if self.producer is not None:
            return True
        if not self.bootstrap_servers or not self.topic:
            return False
        now = time.monotonic()
        if not force and now < self._next_connect_time:
            return False
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                acks='all',
                retries=3,
                retry_backoff_ms=300,
                api_version_auto_timeout_ms=2000,
                max_block_ms=5000,
            )
            self._next_connect_time = 0
            return True
        except Exception as exc:
            self.producer = None
            self._next_connect_time = now + 5
            print(f'failed to create sender: {exc!r}', flush=True)
            return False

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

    def is_none(self):
        return not self.bootstrap_servers or not self.topic

    def get_task_id(self):
        return self.taskId

    def send(self, message_dict):
        if not self._connect():
            # 只保留最新状态；断线贯穿整个任务时，优先保证最终状态能够补发。
            self._pending_message = deepcopy(message_dict)
            return False
        pending_message = self._pending_message
        self._pending_message = None
        if pending_message is not None and not self._send_message(pending_message):
            self._pending_message = pending_message
            return False
        sent = self._send_message(message_dict)
        if not sent and self._pending_message is None:
            self._pending_message = deepcopy(message_dict)
        return sent

    def _send_message(self, message_dict):
        message_dict = deepcopy(message_dict)
        message_dict = self._check_basic_message(message_dict)
        message_dict = self._append_fixed_message(message_dict)
        message_dict = self._build_msg_dict(message_dict)
        msg = json.dumps(message_dict).encode('utf-8')
        for attempt in range(2):
            try:
                self.producer.send(self.topic, msg).get(timeout=5)
                return True
            except Exception as exc:
                print(f'failed to send message: {exc!r}', flush=True)
                self._discard_producer()
                if attempt == 0 and not self._connect(force=True):
                    break
        return False

    def _discard_producer(self):
        producer = self.producer
        self.producer = None
        if producer is None:
            return
        try:
            producer.close(timeout=0)
        except Exception:
            pass

    def close(self):
        pending_message = self._pending_message
        if pending_message is not None:
            self._pending_message = None
            if not self._connect(force=True) or not self._send_message(pending_message):
                self._pending_message = pending_message
                print('failed to deliver pending Kafka message before close', flush=True)
        producer = self.producer
        self.producer = None
        if producer is None:
            return
        try:
            producer.flush(timeout=5)
            producer.close(timeout=2)
        except Exception as exc:
            print(f'failed to close sender: {exc!r}', flush=True)

    def calc_progress_value(self, index, total, min_value=0, max_value=100):
        return int(index / total * (max_value - min_value) + min_value)
