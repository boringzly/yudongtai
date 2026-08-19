import json
import requests
import os
import traceback
#任务状态
post_status = {
    'created': 0, #创建完成(异步)
    'inited': 1,  #初始化完成(异步)
    'running': 2, #正在运行(同步，异步)
    'finish': 3,  #完成(异步)
    'error': 4,   #错误(异步)
    'cancel': 5,  #取消(异步)
}

#url:callback函数的url，由调用者传入
#status：任务状态，使用post_status枚举值
#progress 任务进度 0-100的整数
#feedback_info 需要返回的结构体，包括报错信息和需要返回的非文件类参数------没有需要传入的数据时，传入None
    #errorMessage 报错信息，以json的形式进行组织 例如：{'errorCode':4001 ,'message':'输入文件不存在'}
    #result 需要返回的非文件类参数，以json的形式进行组织 例如：{'outputOtherParams':{{'outGeoArea':123.43}}"
#job_id 任务ID，由调用者传入
def post_progress(url, status, progress, errorMessage, job_id, result = ""):
    payload = json.dumps({'process': progress, 'status': status, 'feedback_info': {'errorMessage': errorMessage,"result": result},'job_id': job_id})
    headers = {
        'user-agent': "vscode-restclient",
        'content-type': "application/json",
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Method': '*',
        'Access-Control-Allow-Headers': '*'
    }
    try:
        response = requests.request("POST", url, data=payload, headers=headers, timeout=5)
        print(response)
        return (response.status_code == 200)
    except Exception as ex:
        print('回调发生异常',ex)
        if status==post_status['finish'] and progress==100:
            post_progress(url, status, progress, errorMessage, job_id,result)
        else:
            return 'error'
#检查文件路径是否存在，不存在则创建路径
def check_file_path(path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        # 如果文件夹不存在，使用os.makedirs()创建它
        os.makedirs(folder_path)
        print(f"文件夹 '{folder_path}' 已创建。")
    else:
        print(f"文件夹 '{folder_path}' 已存在。")
#将报错错误信息写入到日志        
def dump_error_to_file(error_name,exception_object):
    # print(error_name+' error: ',exception_object)
    # traceback.print_exc(file=open(error_name+".log",'w',encoding='utf-8'))
    with open(error_name+".log", "w") as file:
        print(exception_object, file=file)
