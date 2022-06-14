import tensorflow as tf
import numpy as np
import os
import requests

def send_notice(title, content):
    token = "15fe03f24be3487faa7026f09896ebd3"
    url = f"http://www.pushplus.plus/send?token={token}&title={title}&content={content}&template=txt"
    response = requests.request("GET", url)
    #print(response.text)

def GPU(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def GPU_1():
    #tf.config.set_soft_device_placement(True)  # 自动选择一个现有且受支持的设备来运行操作，以避免指定的设备不存在
    #tf.debugging.set_log_device_placement(True)  # 查出我们的操作和张量被配置到哪个 GPU 或 CPU 上
    gpus = tf.config.experimental.list_physical_devices('GPU') # 获取物理GPU个数
    print('物理GPU个数为：', len(gpus))
    print("物理GPU", gpus)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('逻辑GPU个数为：', len(logical_gpus))
    print("逻辑GPU", logical_gpus)
    # 设置内存自增长
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print('args.args.args.args.args.args.-已设置完GPU内存自增长args.args.args.args.args.args.args.')

def print_parameters(model):
    parameters = 0
    for variable in model.trainable_variables:
        parameters += np.product([x for x in variable.get_shape()])
    log = 'TP:{:,}'.format(parameters)  # The number of trainable parameters
    return log

def log(args, Total_time):
    s = "Total Time:{}   ".format(Total_time)
    s += args.Loss_Save_Path
    s += '\n'
    f = open(args.Log_Save_Path,mode='a')
    f.writelines(s)
    f.close()

'''
给出一个路径（单级目录或多级别目录），若它不存在，则创建；若存在，则跳过
'''
def create_path(args):
    dirs = [args.Input_Path, args.Train_Results_Path, args.Train_Results_Figures_Path, args.Train_Results_Models_Path, args.Train_Results_Logs_Path]
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)