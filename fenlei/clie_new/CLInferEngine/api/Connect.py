import numpy as np
import requests
import json
import os
import time
import copy
import math

EARTH_RADIUS = 6378137      # 地球半径

class AiServerProxy:
    def __init__(self):
        # self.__aiServerUrl = os.environ.get('AI_SERVER_URL')
        # self.__aiTaskId = os.environ.get('AI_TASK_ID')
        self.__aiServerUrl = 'http://192.168.100.163:8098/'
        self.__aiTaskId = "2f15d91b08a34e038db5126edea756d9"
        self.headers = {'Content-Type': 'application/json'}
        self.__blockId = 0
        self.__taskInformation = None
        self.originShift = 2 * math.pi * EARTH_RADIUS / 2


        current_time = time.asctime(time.localtime(time.time()))
        log = current_time + ' Class AiServerProxy' + \
              ' AiServerUrl:%s' % self.__aiServerUrl + \
              ' AiTaskId:%s' % self.__aiTaskId

        print(log)

    def taskInfo(self):
        """
        任务开始时调用
        请求任务开始
        """
        req = InfoParament()
        req['taskId'] = self.__aiTaskId
        apiUrl = self.__aiServerUrl + 'api/v1/task/info'
        para = req.getPara()

        current_time = time.asctime(time.localtime(time.time()))
        log = current_time + ' AiServerProxy.taslnfo' + ' task info parament:' + str(para)
        print(log)

        datas = json.dumps(para)
        # r = requests.post(apiUrl, data=datas, headers=self.headers)
        # self.__taskInformation = json.loads(r.text)['data']
        # "path fixed"
        # self.__taskInformation['pathInfo']['preImgPath'] = '/nfs/project/netdisk/192.168.10.227/d/private/dongsj/road_test/data/beijing2.tif'
        # self.__taskInformation['pathInfo']['outputPath'] = '/nfs/project/netdisk/192.168.10.227/d/private/dongsj/road_test/pred'

    def taskPrepare(self, preparePhase: str, progress: int):
        """
        发送预测开始前的相关信息状态
        preparePhase: 当前的任务名称
        progress: 某阶段的任务
        """
        pre = InfoPrepare()
        pre["taskId"] = self.__aiTaskId
        pre["preparePhase"] = preparePhase
        pre["progress"] = progress
        apiUrl = self.__aiServerUrl + 'api/v1/task/prepare'

        para = pre.getPara()

        current_time = time.asctime(time.localtime(time.time()))
        log = current_time + ' AiServerProxy.taskPrepare' + ' task prepare parament:' + str(para)
        print(log)
        datas = json.dumps(para)
        r = requests.post(apiUrl, data=datas, headers=self.headers)

    def taskClear(self, imgRange, slices, estTime):
        """
        发送任务相关信息.
        :param imgRange: 列表存放整个图片的信息，pictureRange[0]最小经度，pictureRange[1]最大经度，pictureRange[2]最小纬度, pictureRange[3]是最大纬度
        :param slices: 二位列表存放每个切片的信息, slice[i][0]切片的最小经度, slice[i][1]切片的最大经度,slice[i][2]切片的最小纬度, slice[i][3]切片的最大纬度
        :param estTime: 预计任务执行时间
        """
        clear = ClearParament()
        clear['taskId'] = self.__aiTaskId
        clear['estTime'] = estTime
        apiUrl = self.__aiServerUrl + '/api/v1/task/clear'

        clear['extend']['minLng'] = imgRange[0]
        clear['extend']['maxLng'] = imgRange[1]
        clear['extend']['minLat'] = imgRange[2]
        clear['extend']['maxLat'] = imgRange[3]

        blockTemp = {
                'blockId': '',
                'extend': {
                    'minLng': 0,
                    'maxLng': 0,
                    'minLat': 0,
                    'maxLat': 0
                }
            }

        for blockId in range(len(slices)):
            blockTemp['blockId'] = str(blockId)
            blockTemp['extend']['minLng'] = slices[blockId][0]
            blockTemp['extend']['maxLng'] = slices[blockId][1]
            blockTemp['extend']['minLat'] = slices[blockId][2]
            blockTemp['extend']['maxLat'] = slices[blockId][3]
            clear['blocks'].append(blockTemp)
        para = clear.getPara()

        current_time = time.asctime(time.localtime(time.time()))
        log = current_time + ' AiServerProxy.taskClear' + ' task clear parament:' + str(para)
        print(log)

        datas = json.dumps(para)
        r = requests.post(apiUrl, data=datas, headers=self.headers)


    def taskClear_2(self, imgRange, clip_geo_1, clip_geo_2, img_size, estTime):
        """
        发送任务相关信息.
        :param imgRange:列表存放整个图片的信息，pictureRange[0]最小经度，pictureRange[1]最大经度，pictureRange[2]最小纬度, pictureRange[3]是最大纬度
        :param clip_geo_1: 水平分辨率
        :param clip_geo_2: 竖直分辨率
        :param img_size: 切片尺寸
        :param estTime: 任务时间
        :return:
        """
        if clip_geo_1 < 0:
            clip_geo_1 = -clip_geo_1
        if clip_geo_2 < 0:
            clip_geo_2 = -clip_geo_2
        clear = ClearParament()
        clear['taskId'] = self.__aiTaskId
        clear['estTime'] = estTime
        apiUrl = self.__aiServerUrl + 'api/v1/task/clear'
        blocks = []
        blockcp = {
            'blockId': '',
            'extend': {
                'minLng': 0,
                'maxLng': 0,
                'minLat': 0,
                'maxLat': 0
            }
        }
        idx = [0, 0]
        for i in np.arange(imgRange[3], imgRange[2], img_size * -clip_geo_2):
            idx[1] = 0
            for j in np.arange(imgRange[0], imgRange[1], img_size * clip_geo_1):
                block = copy.deepcopy(blockcp)
                block['blockId'] = str(idx)
                block['extend']['minLng'] = j
                block['extend']['maxLng'] = j + clip_geo_1 * img_size
                block['extend']['minLat'] = i - clip_geo_2 * img_size
                block['extend']['maxLat'] = i
                if block['extend']['maxLng'] > imgRange[1]:
                    block['extend']['maxLng'] = imgRange[1]
                if block['extend']['minLat'] < imgRange[2]:
                    block['extend']['minLat'] = imgRange[2]
                if imgRange[1] > 180:
                    block['extend']['minLng'] = self.meterToLng(block['extend']['minLng'])
                    block['extend']['maxLng'] = self.meterToLng(block['extend']['maxLng'])
                    block['extend']['minLat'] = self.meterToLat(block['extend']['minLat'])
                    block['extend']['maxLat'] = self.meterToLat(block['extend']['maxLat'])
                blocks.append(block)
                idx[1] += 1
            idx[0] += 1
        clear['blocks'] = blocks
        if imgRange[1] > 180:
            clear['extend']['minLng'] = self.meterToLng(imgRange[0])
            clear['extend']['maxLng'] = self.meterToLng(imgRange[1])
            clear['extend']['minLat'] = self.meterToLat(imgRange[2])
            clear['extend']['maxLat'] = self.meterToLat(imgRange[3])

        else:
            clear['extend']['minLng'] = imgRange[0]
            clear['extend']['maxLng'] = imgRange[1]
            clear['extend']['minLat'] = imgRange[2]
            clear['extend']['maxLat'] = imgRange[3]
        para = clear.getPara()
        current_time = time.asctime(time.localtime(time.time()))
        log = current_time + ' AiServerProxy.taskClear' + ' task clear parament:' + str(para)
        print(log)
        datas = json.dumps(para)
        r = requests.post(apiUrl, data=datas, headers=self.headers)

    def taskProgress(self, geoFeatures):
        """
        每预测完一张切片调用一次
        :param geoFeatures:
        :return:
        """
        updata = ProgressParament()
        apiUrl = self.__aiServerUrl + '/api/v1/task/progress'
        updata['taskId'] = self.__aiTaskId
        updata['blockId'] = str(self.__blockId)
        updata['geoFeatures'] = geoFeatures
        para = updata.getPara()
        current_time = time.asctime(time.localtime(time.time()))
        log = current_time + ' AiServerProxy.taskProgress' + ' task progress parament:' + str(para)
        print(log)
        datas = json.dumps(para)
        r = requests.post(apiUrl, data=datas, headers=self.headers)
        self.__blockId += 1


    def taskProgress_2(self, geoFeatures, blockId):
        if isinstance(blockId, tuple):
            blockId = list(blockId)
        updata = ProgressParament()
        apiUrl = self.__aiServerUrl + 'api/v1/task/progress'
        updata['taskId'] = self.__aiTaskId
        updata['blockId'] = str(blockId)
        if not isinstance(geoFeatures, str):
            geoFeatures = json.dumps(geoFeatures)
        updata['geoFeatures'] = geoFeatures
        para = updata.getPara()
        current_time = time.asctime(time.localtime(time.time()))
        log = current_time + ' AiServerProxy.taskProgress' + ' task progress parament:' + str(para)
        print(log)
        datas = json.dumps(para)
        r = requests.post(apiUrl, data=datas, headers=self.headers)


    def taskFinish(self, shp_name, shapeType):
        """
        预测完成调用
        :param shp_name: 保存shp的文件名
        :param shapeType: 任务类型。传入整型，1：变化检测，2：语义分割，3：目标检测
        :return:
        """
        finish = FinishParament()
        apiUrl = self.__aiServerUrl + 'api/v1/task/finish'
        finish['taskId'] = self.__aiTaskId
        finish['shapeFile'] = self.__taskInformation['pathInfo']['outputPath'] + shp_name
        finish['shapeType'] = shapeType
        para = finish.getPara()
        current_time = time.asctime(time.localtime(time.time()))
        log = current_time + ' AiServerProxy.taskFinish' + ' task finish parament:' + str(para)
        print(log)
        datas = json.dumps(para)
        r = requests.post(apiUrl, data=datas, headers=self.headers)


    def taskFinish_2(self, shp, shapeType=2):
        """
        :param shp: shp文件的绝对路径
        :param shapeType: 任务类型。传入整型，1：变化检测，2：语义分割，3：目标检测
        :return:
        """
        finish = FinishParament()
        finish['taskId'] = self.__aiTaskId
        apiUrl = self.__aiServerUrl + 'api/v1/task/finish'
        finish['shapeFile'] = shp
        finish['shapeType'] = shapeType
        para = finish.getPara()
        current_time = time.asctime(time.localtime(time.time()))
        log = current_time + ' AiServerProxy.taskFinish' + ' task finish parament:' + str(para)
        print(log)
        datas = json.dumps(para)
        r = requests.post(apiUrl, data=datas, headers=self.headers)

    def getInfo(self):
        """
        获取任务信息, 返回一个字典
        :return: 返回任务参数信息
        """
        if self.__taskInformation != None:
            return self.__taskInformation
        else:
            print("任务信息为空")
            return None

    def meterToLng(self, m_Lng):
        Lng = (m_Lng / self.originShift) * 180
        return Lng

    def meterToLat(self, m_lat):
        lat = (m_lat / self.originShift) * 180
        lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
        return lat


class InfoPrepare:
    def __init__(self):
        self.__para = {"taskId": None,
                       "preparePhase": None,
                       "progress": None}

    def __getitem__(self, key):
        return self.__para[key]

    def __setitem__(self, key, value):
        self.__para[key] = value

    def getPara(self):
        return self.__para


class InfoParament:
    def __init__(self):
        self.__para = {'taskId': None}

    def __getitem__(self, key):
        return self.__para[key]

    def __setitem__(self, key, value):
        self.__para[key] = value

    def getPara(self):
        return self.__para



class ClearParament:
    def __init__(self):
        self.__para = {'taskId': None,
                       'extend': {
                           'minLng': None,
                           'maxLng': None,
                           'minLat': None,
                           'maxLat': None
                       },
                       'estTime': None,
                       'blocks': []
                       }

    def __getitem__(self, key):
        return self.__para[key]

    def __setitem__(self, key, value):
        self.__para[key] = value

    def getPara(self):
        return self.__para


class ProgressParament:
    def __init__(self):
        self.__para = {'taskId': None,
                       'blockId': None,
                       'duration': 0,
                       'status': 1,
                       'geoFeatures': {}}

    def __getitem__(self, key):
        return self.__para[key]

    def __setitem__(self, key, value):
        self.__para[key] = value

    def getPara(self):
        return self.__para


class FinishParament:
    def __init__(self):
        self.__para = {'taskId': None,
                       'shapeFile': None,
                       'shapeType': None}

    def __getitem__(self, key):
        return self.__para[key]

    def __setitem__(self, key, value):
        self.__para[key] = value

    def getPara(self):
        return self.__para


def get_block_data(imgRange, clip_geo_1, clip_geo_2, img_size, estTime):
    """
    发送任务相关信息.
    :param imgRange:列表存放整个图片的信息，pictureRange[0]最小经度，pictureRange[1]最大经度，pictureRange[2]最小纬度, pictureRange[3]是最大纬度
    :param clip_geo_1: 水平分辨率
    :param clip_geo_2: 竖直分辨率
    :param img_size: 切片尺寸
    :param estTime: 任务时间
    :return:
    """
    if clip_geo_1 < 0:
        clip_geo_1 = -clip_geo_1
    if clip_geo_2 < 0:
        clip_geo_2 = -clip_geo_2
    originShift = 2 * math.pi * EARTH_RADIUS / 2
    clear = ClearParament()
    clear['estTime'] = estTime
    blocks = []
    blockcp = {
        'blockId': '',
        'extend': {
            'minLng': 0,
            'maxLng': 0,
            'minLat': 0,
            'maxLat': 0
        }
    }
    idx = [0, 0]

    def meterToLng(m_Lng):
        Lng = (m_Lng / originShift) * 180
        return Lng
    
    def meterToLat(m_lat):
        lat = (m_lat / originShift) * 180
        lat = 180 / math.pi * (2 * math.atan(math.exp(lat * math.pi / 180)) - math.pi / 2)
        return lat

    for i in np.arange(imgRange[3], imgRange[2], img_size * -clip_geo_2):
        idx[1] = 0
        for j in np.arange(imgRange[0], imgRange[1], img_size * clip_geo_1):
            block = copy.deepcopy(blockcp)
            block['blockId'] = str(idx[0]) + '_' + str(idx[1])
            block['extend']['minLng'] = j
            block['extend']['maxLng'] = j + clip_geo_1 * img_size
            block['extend']['minLat'] = i - clip_geo_2 * img_size
            block['extend']['maxLat'] = i
            if block['extend']['maxLng'] > imgRange[1]:
                block['extend']['maxLng'] = imgRange[1]
            if block['extend']['minLat'] < imgRange[2]:
                block['extend']['minLat'] = imgRange[2]
            if imgRange[1] > 180:
                block['extend']['minLng'] = meterToLng(block['extend']['minLng'])
                block['extend']['maxLng'] = meterToLng(block['extend']['maxLng'])
                block['extend']['minLat'] = meterToLat(block['extend']['minLat'])
                block['extend']['maxLat'] = meterToLat(block['extend']['maxLat'])
            blocks.append(block)
            idx[1] += 1
        idx[0] += 1
    clear['blocks'] = blocks
    if imgRange[1] > 180:
        clear['extend']['minLng'] = meterToLng(imgRange[0])
        clear['extend']['maxLng'] = meterToLng(imgRange[1])
        clear['extend']['minLat'] = meterToLat(imgRange[2])
        clear['extend']['maxLat'] = meterToLat(imgRange[3])

    else:
        clear['extend']['minLng'] = imgRange[0]
        clear['extend']['maxLng'] = imgRange[1]
        clear['extend']['minLat'] = imgRange[2]
        clear['extend']['maxLat'] = imgRange[3]
    para = clear.getPara()
    # current_time = time.asctime(time.localtime(time.time()))
    # log = current_time + ' AiServerProxy.taskClear' + ' task clear parament:' + str(para)
    # print(log)
    datas = json.dumps(para)
    return datas

