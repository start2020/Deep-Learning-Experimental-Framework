from libs import para, utils
import argparse
import numpy as np
import datetime
import argparse
from libs import para
import inspect

# 变量名转字符串
def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

# Data Split = Train Data + Test Data
def data_split_save(Data, Data_Name, fp):
    M = Data[0].shape[0]
    Train_Num = int(args.Train_Ratio * M)
    for i in range(len(Data)):
        train = Data[i][:Train_Num]
        test = Data[i][Train_Num:]
        # Save Data: (DM, ...)
        path = args.Save_Path + Data_Name[i] + '.npz'
        np.savez_compressed(path, train, test)
        print("{} => dtype:{}, train_shape:{}, test_shape:{}".format(Data_Name[i], train.dtype, train.shape, test.shape), file=fp)

def spatiotemporal(D, N, TT):
    M, P = TT.shape
    n = np.array([j for j in range(N)])  # (N,)
    os = np.reshape(n, newshape=(1, 1, 1, N, 1, 1))  # (1,1,1,N_o,1,1)
    os = np.tile(os, (M,D,P,1,N,1))  # (M,D,P,N_o,N_d,1)
    ds = np.reshape(n, newshape=(1, 1, 1, 1, N, 1))  # (1,1,1,1,N_d,1)
    ds = np.tile(ds, (M,D,P,N,1,1))  # (M,D,P,N_o,N_d,1)

    ws = np.array([d % 7 for d in range(D)])  # (D,)
    ws = np.reshape(ws, newshape=(1,D, 1, 1, 1, 1))  # (1,D,1,1,1,1)
    ws = np.tile(ws, (M,1,P,N,N,1))  # (M,D,P,N_o,N_d,1)

    ts = np.reshape(TT, newshape=(M, 1, P, 1, 1, 1))  # (M,P)=>(M,1,P,1,1,1)
    ts = np.tile(ts, (1,D,1,N,N,1)) # (M,D,P,N_o,N_d,1)
    ST = np.concatenate([os, ds, ws, ts], axis=-1)  # (M,D,P,N_o,N_d,4)
    return ST


def generate_samples(args,fp):
    print("Loading Data", file=fp)
    start_time = datetime.datetime.now() # 开始记录时间
    OD = np.load(args.Data_Path)['matrix'] #(D,T,N,N,T)
    D, T, N = OD.shape[0], OD.shape[1], OD.shape[2]
    # D, T, N = 21, 30, 5
    # OD = np.random.randint(0,8,size=(D,T,N,N,T))
    P = args.P
    print("Data shape:{}, Datatype:{}".format(OD.shape, OD.dtype), file=fp)
    end_time_1 = datetime.datetime.now()
    print("Done! Loading Time:{}\n".format(end_time_1-start_time), file=fp)

    print("Generate Samples",file=fp)
    # generate samples
    OtD_b, OtD, ODt, Label,TT = [], [], [], [], []
    Vs = [OtD_b, OtD, ODt, Label,TT]
    for t in range(T): # t is the predicted slot
        if t - P <= 0:
            continue
        x_OtD_b = np.sum(OD[:,t-P:t,:,:,:t], axis = -1) #(D, P, N_o, N_d) t时刻前
        x_OtD = np.sum(OD[:,t-P:t,:,:,:], axis = -1) #(D, P, N_o, N_d) 前P个完整的OD矩阵
        x_ODt = np.sum(OD[...,t-P:t], axis = 1) #(D, 1+T, N_o, N_d, T+1)=>(D, N_o, N_d, P) 出流矩阵
        label = np.sum(OD[:,t,...], axis = -1) #(D, N_o, N_d) t时刻的目标值
        ts = np.array([i for i in range(t - P, t)])  # (P,) # 时刻值
        vs = [x_OtD_b, x_OtD, x_ODt, label,ts]
        for v, V in zip(vs, Vs):
            V.append(v)
    # stack
    for i in range(len(Vs)):
        Vs[i] = np.stack(Vs[i], axis=0)
    OtD_b, OtD, ODt, Label, TT = Vs #TT(M,P)
    ST =spatiotemporal(D, N, TT) # (M, D, P, N_o, N_d,4)
    del TT

    OtD_W = OtD[:, 0:-7]  # (M, D-7, P, N_o, N_d)t时刻前，往后去掉一周
    OtD_b_W = OtD_b[:, 0:-7]  # (M, D-7, P, N_o, N_d)t时刻前，往后去掉一周
    OtD_a_W = OtD_W - OtD_b_W
    OtD_W[OtD_W == 0] = 1  # 0的位置用1代替，防止除零错误
    OtD_d_W_P = (OtD_a_W / OtD_W).astype(np.float32) # historical delayed OD probability matrix
    del OtD_W, OtD_b_W, OtD_a_W

    Label_W = Label[:, 0:-7]  # (M, D-7, N_o, N_d) 目标的前一周同一个时刻
    Label_M = np.stack([Label[:, i - 7:i] for i in range(7, D)], axis=1)  # (M,D-7,7,N_o, N_d) # 前M=7天的同一个时刻

    ST = ST[:, 7:] # (M, D-7, P, N_o, N_d) 去掉最开始的一周
    OtD = OtD[:, 7:]  # (M, D-7, N_o, N_d,P) 去掉最开始的一周
    OtD_b = OtD_b[:, 7:]  # (M, D-7, P, N_o, N_d) 去掉最开始的一周
    ODt = ODt[:, 7:]  # (M, D-7, N_o, N_d,P) 去掉最开始的一周
    ODt = np.transpose(ODt, (0, 1, 4, 2, 3))  # (M, D-7, N_o, N_d, P) => (M, D-7,P, N_o, N_d)
    Label = Label[:,7:]  # (M, D-7, N_o, N_d) 去掉最开始的一周

    Vs = [OtD_b,OtD, ODt, OtD_d_W_P, Label, Label_W, Label_M, ST]
    for i in range(len(Vs)):
        # (M,D-7)=>(1,M(D-7),...)=>(M(D-7)...)
        Vs[i] = np.squeeze(np.concatenate(np.vsplit(Vs[i], Vs[i].shape[0]), axis=1), axis=0)
    OtD_b,OtD, ODt, OtD_d_W_P, Label, Label_W, Label_M, ST = Vs

    mean,std = np.mean(OtD), np.std(OtD)
    train_mean = np.mean(OtD[:int(args.Train_Ratio * OtD.shape[0])])
    train_std = np.std(OtD[:int(args.Train_Ratio * OtD.shape[0])])
    train_mean_b = np.mean(OtD_b[:int(args.Train_Ratio * OtD_b.shape[0])])
    train_std_b = np.std(OtD_b[:int(args.Train_Ratio * OtD_b.shape[0])])
    print("OtD => mean:{:.2f},std:{:.2f}".format(mean, std), file=fp)
    print("OtD_train => train_mean:{:.2f},train_std:{:.2f}".format(train_mean, train_std), file=fp)
    print("OtD_b => train_mean_b:{:.2f},train_std_b:{:.2f}".format(train_mean_b, train_std_b), file=fp)
    Min, Max, Ptp = np.min(OtD), np.max(OtD), np.ptp(OtD)
    print("OtD=>Min:{},Max:{},Ptp:{}".format(Min, Max,Ptp),file=fp)

    #mean, std = 0, 1

    np.savez_compressed(args.Mean_Std_Path, mean=mean, std=std) # 保存好std,mean
    OtD_b = ((OtD_b-mean)/std).astype(np.float32) # 处理输入值，经过标准化
    OtD = ((OtD-mean)/std).astype(np.float32)# 处理输入值，经过标准化
    ODt = ((ODt-mean)/std).astype(np.float32) # 处理输入值，经过标准化

    Inflow = np.sum(OtD, axis=-1) #((D-7)M, P, N_o)  【输入 CASCNN、AHGCSP补全】
    Finished_Inflow = np.sum(OtD_b, axis=-1) #((D-7)M, P, N_o,)
    Delayed_Inflow = Inflow - Finished_Inflow #((D-7)M, P, N_o,) 【输入 AHGCSP补全】
    del Finished_Inflow
    Outflow = np.sum(ODt, axis=2) # ((D-7)M, P, N_o, N_d) => ((D-7)M, P, N_d) 【输入 CASCNN】

    end_time_2 = datetime.datetime.now()
    print("Generate Samples Time:{}\n".format(end_time_2-end_time_1), file=fp)

    # 变量名转字符串
    Data = [OtD_b, ODt, OtD_d_W_P, Label, Label_W, Label_M, ST, Inflow, Delayed_Inflow, Outflow]
    Data_Name = []
    for i in range(len(Data)):
        Data_Name.append(retrieve_name(Data[i])[0])

    print("Save Smaples", file=fp)
    data_split_save(Data, Data_Name, fp)
    end_time_3 = datetime.datetime.now()
    print("Save Smaples Time:{}\n".format(end_time_3-end_time_2), file=fp)

'''
HA/GCN/GRU/ConvLSTM: 当天前p个时刻的不全OtD矩阵 => OtD_b
CASCNN: 当前天p个时刻的Inflow和Outflow向量 + 前M天同一个时刻的OtD矩阵 => Outflow + Inflow + ODt_M
GEML: 当天前p个时刻的不全OtD矩阵 + 预测的Inflow/Outflow => OtD_b + 目标矩阵产生就可以
AHGCSP:  当天前p个时刻的不全OtD矩阵 + 当天前p个时刻的ODt矩阵 + 目标的前一周的同一个时刻 => OtD_b + 
          + 当前天p个时刻的Inflow +  ST=> Label_W + Inflow
All Models: 需要同样的预测目标 => Label
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    para.Parameter(parser)
    args = parser.parse_args()
    # 生成该数据集下的所有目录
    utils.create_path(args)
    # 打开日志文件
    fp = open(args.Log_Inputs_Path, "a+")  # a+ 如果文件不存在就创建。存在就在文件内容的后面继续追加
    print("############################################################", file=fp)
    print("Begin!", file=fp)

    info = ["City:{}".format(args.City), "T:{}".format(args.T), "P:{}".format(args.P), "Data:{}".format(args.Dataset)]
    info = " ".join(info)
    print(info, file=fp)
    try:
        generate_samples(args,fp)
    except Exception as err:
        title = "Sample Generate Error!"
        content = str(err) + '\n' + info
        utils.send_notice(title, content) # content f"训练正确率:55%\n测试正确率:96.5%"
        print(err, file=fp)
    print("Done!\n",file=fp)
    fp.close()

'''
关于减小存储空间/数据装入时间/数据一致性问题:
(1)产生数据时，如果能尽量小的存储，就尽量小；
(2)如果要浮点数，则不能用默认的浮点数float64，要用float32。
(3)如果要整数，则不能用默认的整数int64，要用int32。但服务器上默认的数据居然是int64，不知道什么时候改的。
(4)要强制性地设置数据格式
'''
'''
HA
Data shape:  (28, 65, 80, 80, 65)  Datatype:  int32
Done! Loading Time:  0:00:05.427609
OtD => mean:3.10,std:6.68
OtD => train_mean:3.24,train_std:7.05
OtD_b => train_mean_b:2.67,train_std_b:6.93
Done! Spent Time:  0:00:54.725944

SH
Data shape:  (30, 65, 288, 288, 65)  Datatype:  int32
Done! Loading Time:  0:01:21.504462
OtD => mean:1.28,std:2.88
OtD => train_mean:1.32,train_std:3.04
OtD_b => train_mean_b:0.68,train_std_b:3.01
Done! Spent Time:  0:12:37.598710
Total Time:0:28:30.992317

SZ
Data shape:  (30, 65, 118, 118, 65)  Datatype:  int32
Done! Loading Time:  0:00:11.661518
OtD => mean:2.03,std:5.46
OtD => train_mean:2.04,train_std:5.59
OtD => train_mean_b:1.42,train_std_b:5.57
Done! Spent Time:  0:01:10.119727
Total Time:0:03:11.329597

'''

'''
(1)HZ
HA: MAE:1.1936   RMSE:2.4972  WMAPE:0.7238 SMAPE:0.3686
ANN: MAE:1.032     RMSE:2.2699   WMAPE:0.6258  SMAPE:0.3054
前一天同一个时刻：{'MAE': 1.6324, 'RMSE': 3.6243, 'WMAPE': 0.6556, 'SMAPE': 0.4034}
x (1512, 80, 80) y (1512, 80, 80)
前一周的同一个时刻：{'MAE': 1.5466486, 'RMSE': 3.4607, 'WMAPE': 0.6134, 'SMAPE': 0.3922}  
x (1176, 80, 80) y (1176, 80, 80)

(2)SH
HA: MAE:0.3658   RMSE:1.1169  WMAPE:1.0414 SMAPE:0.1698
ANN: MAE:0.3038    RMSE:1.0511   WMAPE:0.8649  SMAPE:0.1299 
前一天同一个时刻：{'MAE': 0.5552, 'RMSE': 1.5702, 'WMAPE': 0.8673, 'SMAPE': 0.2089}
x (1624, 288, 288) y (1624, 288, 288)
前一周的同一个时刻：{'MAE': 0.5260, 'RMSE': 1.3843, 'WMAPE': 0.8029, 'SMAPE': 0.2033}
x (1288, 288, 288) y (1288, 288, 288)
'''
