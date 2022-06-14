#!/usr/bin/env python
# coding: utf-8

import datetime

def Parameter(parser):
    parser.add_argument('--Model', default="ConvLSTM")  # "HA","GEML", "GCN", "CASCNN", "ConvLSTM", "GRU"
    parser.add_argument('--Note', default="OtD_standarize", type=str, help="Some Notice")
    parser.add_argument('--LR', default=0.01, type=float, help="Learning_Rate")
    parser.add_argument('--Batch', default=16, type=int)
    parser.add_argument('--Epochs', default = 1000, type=int)
    parser.add_argument('--GPU', default="0", type=str)
    parser.add_argument('--Repeat', default = 3, type=int)
    parser.add_argument('--City', default = 'HZ', type=str)
    parser.add_argument('--Dataset', default='OD15_01.npz')
    parser.add_argument('--T', default = 15, type=int, help="Time Granularity")
    parser.add_argument('--P', default = 8, type=int, help="Input Steps")
    parser.add_argument('--ANN_Units', default="128,128", type=str)
    parser.add_argument('--ANN_Activations', default="relu,relu", type=str)

    parser.add_argument('--GRU_Units', default="64,64", type=str)

    parser.add_argument('--GCN_Units', default="128, 128", type=str)
    parser.add_argument('--GCN_Activations', default="relu,relu", type=str)

    parser.add_argument('--AHGCSP_GCN_Units', default="128", type=str)
    parser.add_argument('--AHGCSP_Dynamic_Units', default="32", type=str)
    parser.add_argument('--AHGCSP_T_Units', default="32", type=str)
    parser.add_argument('--AHGCSP_S_Units', default="32", type=str)

    parser.add_argument('--GEML_GCN_Units', default="128,128", type=str)
    parser.add_argument('--GEML_LSTM_Units', default="128", type=str)
    parser.add_argument('--GEML_Weights', default="0.5,0.25,0.25", type=str)

    parser.add_argument('--CASCNN_C', default="1", type=str)
    parser.add_argument('--CASCNN_R', default="1", type=str)

    args = parser.parse_args()
    hyper_para  = ["Repeat:"+str(args.Repeat), args.Note, "LR:"+str(args.LR), "Batch:"+str(args.Batch), "Dataset"+args.Dataset]
    if args.Model == "ANN":
        hyper_para.append("ANN_Units:"+args.ANN_Units)
        hyper_para.append("ANN_Activations:"+args.ANN_Activations)
    elif args.Model == "GRU":
        hyper_para.append("GRU_Units:"+args.GRU_Units)
    elif args.Model == "GCN":
        hyper_para.append("GCN_Units"+args.GCN_Units)
        hyper_para.append("GCN_Activations:"+args.GCN_Activations)
    elif args.Model == "AHGCSP":
        hyper_para.append("AHGCSP_GCN_Units:"+args.AHGCSP_GCN_Units)
        hyper_para.append("AHGCSP_Dynamic_Units:"+args.AHGCSP_Dynamic_Units)
        hyper_para.append("AHGCSP_T_Units:"+args.AHGCSP_T_Units)
        hyper_para.append("AHGCSP_S_Units:"+args.AHGCSP_S_Units)
    elif args.Model == "GEML":
        hyper_para.append("GEML_GCN_Units:"+args.GEML_GCN_Units)
        hyper_para.append("GEML_LSTM_Units:" +args.GEML_LSTM_Units)
        hyper_para.append("GEML_Weights:" + args.GEML_Weights)
    elif args.Model == "CASCNN":
        hyper_para.append("CASCNN_C:" + args.CASCNN_C)
        hyper_para.append(args.CASCNN_R)
    else:
        print(args.Model)
    # 可以传入训练结果中的参数
    parser.add_argument('--hyper_para', default = " ".join(hyper_para), type=str)
    # 可以在显示台打印的参数
    hyper_para.extend([args.City, args.Model])
    print_parameters = " ".join(hyper_para)

    # Unchange
    args = parser.parse_args()
    if args.City == "SH":
        parser.add_argument('--N', default=288, type=int)
    elif args.City == "SZ":
        parser.add_argument('--N', default=118, type=int)
    elif args.City == "HZ":
        parser.add_argument('--N', default=80, type=int)
    else:
        print("No such Dataset!")
    time = datetime.datetime.strftime(datetime.datetime.now(), "%m%d_%H%M")
    parser.add_argument('--Time', default = time, type=str)
    parser.add_argument('--Train_Ratio', default = 0.8, type=float, help ="The ratio of Train Data and Val Data")
    parser.add_argument('--Total_Patience', default = 15, type=int)
    parser.add_argument('--Decay_Patience', default = 10, type=int)
    parser.add_argument('--Decay_Ratio', default = 0.9, type=float)

    # 目录
    args = parser.parse_args()
    parser.add_argument('--Original_Path', default= 'Data/Original/' +args.City + '/', type=str)
    parser.add_argument('--Input_Path', default='Data/Inputs/' + args.City +  '/', type=str)
    parser.add_argument('--Train_Results_Path', default='Data/Train_Results/' + args.City +  '/', type=str)
    parser.add_argument('--Train_Results_Figures_Path', default='Data/Train_Results/' + args.City +  '/Figures/', type=str)
    parser.add_argument('--Train_Results_Models_Path', default='Data/Train_Results/' + args.City +  '/Models/', type=str)
    parser.add_argument('--Train_Results_Logs_Path', default='Data/Train_Results/' + args.City +  '/Logs/', type=str)

    # Path
    args = parser.parse_args()
    parser.add_argument('--Data_Path', default= args.Original_Path + args.Dataset, type=str)
    parser.add_argument('--Save_Path', default= args.Input_Path + 'P' + str(args.P) + 'T' + str(args.T) + args.Dataset.split('.')[0] + '-', type=str)
    parser.add_argument('--Mean_Std_Path', default= args.Input_Path + 'T' + str(args.T) + args.Dataset.split('.')[0] + "_Mean_Std.npz" , type=str)
    parser.add_argument('--GCN_A_Path', default= args.Input_Path + 'GCN_A.npz', type=str)
    parser.add_argument('--GEML_Geo_Path', default= args.Input_Path + 'GEML_Geo.npz', type=str)
    parser.add_argument('--AHGCSP_Geo_Path', default= args.Input_Path + 'AHGCSP_Geo.npz', type=str)
    parser.add_argument('--AHGCSP_KL_Path', default= args.Input_Path + 'AHGCSP_KL.npz', type=str)
    parser.add_argument('--AHGCSP_S_Path', default= args.Input_Path + 'AHGCSP_S.npz', type=str)
    parser.add_argument('--AHGCSP_D_Path', default= args.Input_Path + 'AHGCSP_D.npz', type=str)
    parser.add_argument('--AHGCSP_W_Path', default= args.Input_Path + 'AHGCSP_W.npz', type=str)

    parser.add_argument('--Model_Save_Path', default=args.Train_Results_Models_Path + 'P' + str(args.P) + '_T' + str(
        args.T) + '_' + args.Model + '_' + args.Time + '.h5', type=str)
    parser.add_argument('--Loss_Save_Path', default=args.Train_Results_Figures_Path + 'P' + str(args.P) + '_T' + str(
        args.T) + '_' + args.Model + '_' + args.Time, type=str)
    parser.add_argument('--Metrics_Save_Path', default=args.Train_Results_Path + 'P' + str(args.P) + '_T' + str(
        args.T) + '_' + args.Model + '.txt', type=str)
    parser.add_argument('--Log_Inputs_Path', default=args.Train_Results_Logs_Path + 'log_inputs.txt', type=str)
    parser.add_argument('--Log_Main_Path', default=args.Train_Results_Logs_Path + 'log_main_{}_{}.txt'.format(args.Model, args.Time), type=str)
    return print_parameters
