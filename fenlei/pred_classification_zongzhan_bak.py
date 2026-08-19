import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

from QWEN_CD import main_classification


# encoding: utf-8
import logging
import time
import os
os.environ["PROJ_LIB"] = "/home/luban/.local/lib/python3.8/site-packages/osgeo/data/proj"
import traceback
from logging import handlers
import flask
import requests
import requests_mock

from flask import Flask, jsonify, make_response, request
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

from schedule import post_status, post_progress


plt.switch_backend('agg')

thread_pool_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="test_")
app = Flask(__name__)


def run_one_image_change_detection(input_path1, input_path2, mask_path, output_path, callback_url, job_id):
    try:
        logger.info('接收到url和job_id')
        if not os.path.exists(input_path1) or not os.path.exists(input_path2):
            post_progress(callback_url, post_status['输入文件不存在'], 4001, None, job_id)
            logger.info('输入文件不存在')

        # print(output_path)
        # if not output_path:
        #     post_progress(callback_url, post_status['缺少输出路径'], 400301, None, job_id)
        #     logger.info('缺少输出路径')

        main_classification(input_path1, input_path2, mask_path, output_path, logger=logger, callback_url=callback_url, job_id=job_id)

        post_progress(callback_url, post_status['finish'], 100, None, job_id)
    except Exception as e:
        print(str(e))
        error_message = {"errorCode": 4005, "message": "任务运行中报错line48"+e}
        post_progress(callback_url, post_status['error'], -1, error_message, job_id)


#异步运行
@app.route('/cal_Class_Prediction/', methods=['POST'])
def run_class_prediction():
    data = request.json
    print(data)
    input_path1 = data.get("input_path1")
    input_path2 = data.get("input_path2")
    mask_path = data.get("mask_path")
    output_path = data.get("output_path")
    callback_url = data.get("callback_url")
    job_id = data.get("job_id")

    '''
    input_path1 = 'asset/H49D003012.tif'
    input_path2 = 'asset/H49D003012.tif'
    mask_path = 'asset/Export_Output_2.shp'
    output_path = 'asset/Export_Output_2_result2.shp'
    callback_url = "http:localhost"
    job_id = 1234'''

    logger.info(f'前时相影像路径：{input_path1}')
    logger.info(f'后时相影像路径：{input_path2}')
    logger.info(f'变化掩码路径：{output_path}')
    logger.info(f'类别掩码路径：{output_path}')
    
    try:
        thread_pool_executor.submit(run_one_image_change_detection, input_path1, input_path2, mask_path, output_path, callback_url, job_id)
    except Exception as e:
        print("error", e)
        error_message ={"errorCode": 4005, "message": "任务运行中报错line86"}
        post_progress(callback_url, post_status['error'], -1, error_message, job_id)
    print("success 200")
    return "success", 200


if __name__ == '__main__':

    # 初始化logger

    log_file = "./server_log"
    logger = logging.getLogger('mylog')
    logger.setLevel(logging.INFO)

    file_handlers = logging.FileHandler(log_file)
    file_handlers.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    # th = handlers.TimedRotatingFileHandler(filename=log_file, when='D', backupCount=30,
    #                                        encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器

    fmt1 = '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
    fmt = logging.Formatter(fmt1)
    # th.setFormatter(fmt1)

    ###
    file_handlers.setFormatter(fmt)
    console_handler.setFormatter(fmt)

    # logger.addHandler(th)
    logger.addHandler(file_handlers)
    logger.addHandler(console_handler)

    ################################################
    logger.info('Server starting...')
    ###################
    CORS(app, supports_credentials=True)
    #run_class_prediction()

    try:
        WSGIServer(('0.0.0.0', 5003), app).serve_forever()
    except KeyboardInterrupt:
        exit(0)
    except Exception as e:
        print(e)
        exit(1)


