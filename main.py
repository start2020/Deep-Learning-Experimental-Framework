#!/usr/bin/env python
# coding: utf-8
import argparse
from libs import para, Learning, utils
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    print_parameter = para.Parameter(parser)
    args = parser.parse_args()
    utils.create_path(args)  # 生成该数据集下的所有目录
    utils.GPU(args)
    # f = open(args.Log_Main_Path, 'a')  # a.log 或者a.txt都能够
    # sys.stdout = f
    # sys.stderr = f
    print("############################################################")
    print("Begin!")
    print(print_parameter)  # 在控制台打印参数
    print(args)  # 打印所有参数

    for i in range(args.Repeat):
        if args.Model not in ['HA']:
            Learning.train(args, i)
        Learning.predict(args)
    print("End!" + '\n')

    # try:
    #     for i in range(args.Repeat):
    #         if args.Model not in ['HA']:
    #             Learning.train(args, i)
    #         Learning.predict(args)
    #     print("End!"+'\n')
    # except Exception as err:
    #     title = "Error!"
    #     content = str(err) + '\n' + print_parameter
    #     utils.send_notice(title, content) # content f"训练正确率:55%\n测试正确率:96.5%"
    #     print(err)
    # f.close()