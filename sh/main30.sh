cd /root/yjx/RUN-ALL-0609/
for((i=1;i<=1;i+=1))
do
    #tfpython main.py --GPU 0 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City HZ --T 30 --Dataset OD30_01.npz  --Model HA  --LR 0.01
    #tfpython main.py --GPU 0 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City HZ --T 30 --Dataset OD30_01.npz  --Model ANN   --LR 0.01 --ANN_Units 128,128 --ANN_Activations relu,relu
    #tfpython main.py --GPU 0 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City HZ --T 30 --Dataset OD30_01.npz  --Model GRU   --LR 0.01 --GRU_Units 128,128
    #tfpython main.py --GPU 0 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City HZ --T 30 --Dataset OD30_01.npz  --Model GCN   --LR 0.01 --GCN_Units 128,128 --GCN_Activations relu,relu
    #tfpython main.py --GPU 0 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City HZ --T 30 --Dataset OD30_01.npz  --Model ConvLSTM  --LR 0.01
    #tfpython main.py --GPU 0 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City HZ --T 30 --Dataset OD30_01.npz  --Model CASCNN --LR 0.01 --CASCNN_C 1 --CASCNN_R 1
    #tfpython main.py --GPU 0 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City HZ --T 30 --Dataset OD30_01.npz  --Model GEML  --LR 0.01 --GEML_GCN_Units 128,128  --GEML_LSTM_Units 128  --GEML_Weights 0.5,0.25,0.25
    #tfpython main.py --GPU 0 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City HZ --T 30 --Dataset OD30_01.npz  --Model AHGCSP --LR 0.01 --AHGCSP_GCN_Units 128  --AHGCSP_Dynamic_Units 32  --AHGCSP_T_Units 32  --AHGCSP_S_Units 32

    #tfpython main.py --GPU 1 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SH --T 30 --Dataset OD30_04.npz  --Model HA  --LR 0.01
    #tfpython main.py --GPU 1 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SH --T 30 --Dataset OD30_04.npz  --Model ANN   --LR 0.01 --ANN_Units 64,64 --ANN_Activations relu,relu
    #tfpython main.py --GPU 1 --Note " "  --Repeat 5   --Batch 1  --Epochs 1000 --City SH --T 30 --Dataset OD30_04.npz  --Model GRU   --LR 0.01 --GRU_Units 64,64
    #tfpython main.py --GPU 1 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SH --T 30 --Dataset OD30_04.npz  --Model GCN   --LR 0.01 --GCN_Units 128,128 --GCN_Activations relu,relu
    #tfpython main.py --GPU 1 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SH --T 30 --Dataset OD30_04.npz  --Model ConvLSTM   --LR 0.001
    #tfpython main.py --GPU 1 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SH --T 30 --Dataset OD30_04.npz  --Model CASCNN --LR 0.01 --CASCNN_C 1 --CASCNN_R 1
    #tfpython main.py --GPU 1 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SH --T 30 --Dataset OD30_04.npz  --Model GEML   --LR 0.01 --GEML_GCN_Units 128,128  --GEML_LSTM_Units 128  --GEML_Weights 0.5,0.25,0.25
    #tfpython main.py --GPU 1 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SH --T 30 --Dataset OD30_04.npz  --Model AHGCSP --LR 0.01 --AHGCSP_GCN_Units 128  --AHGCSP_Dynamic_Units 32  --AHGCSP_T_Units 32  --AHGCSP_S_Units 32

    #tfpython main.py --GPU 3 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SZ --T 30 --Dataset OD30_05.npz  --Model HA --LR 0.01
    #tfpython main.py --GPU 3 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SZ --T 30 --Dataset OD30_05.npz  --Model ANN --LR 0.001 --ANN_Units 32,32 --ANN_Activations relu,relu
    #tfpython main.py --GPU 3 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SZ --T 30 --Dataset OD30_05.npz  --Model GRU  --LR 0.01 --GRU_Units 32,32
    #tfpython main.py --GPU 3 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SZ --T 30 --Dataset OD30_05.npz  --Model GCN  --LR 0.01 --GCN_Units 128,128 --GCN_Activations relu,relu
    #tfpython main.py --GPU 3 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SZ --T 30 --Dataset OD30_05.npz  --Model ConvLSTM  --LR 0.001
    #tfpython main.py --GPU 3 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SZ --T 30 --Dataset OD30_05.npz  --Model CASCNN --LR 0.01 --CASCNN_C 1 --CASCNN_R 1
    #tfpython main.py --GPU 3 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SZ --T 30 --Dataset OD30_05.npz  --Model GEML --LR 0.01 --GEML_GCN_Units 128,128  --GEML_LSTM_Units 128  --GEML_Weights 0.5,0.25,0.25
    #tfpython main.py --GPU 3 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City SZ --T 30 --Dataset OD30_05.npz  --Model AHGCSP --LR 0.01 --AHGCSP_GCN_Units 128  --AHGCSP_Dynamic_Units 32  --AHGCSP_T_Units 32  --AHGCSP_S_Units 32

done
pwd
