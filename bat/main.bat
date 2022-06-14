cd ../
for /L %%i in (1,1,1) do (
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 0 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City HZ --T 15 --Dataset OD15_01.npz  --Model HA  --LR 0.01
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 0 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City HZ --T 15 --Dataset OD15_01.npz  --Model ANN   --LR 0.01 --ANN_Units 128,128,1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 0 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City HZ --T 15 --Dataset OD15_01.npz  --Model GRU   --LR 0.01 --GRU_Units 64,64,1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 0 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City HZ --T 15 --Dataset OD15_01.npz  --Model GCN   --LR 0.01 --GCN_Units 128,128,1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 0 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City HZ --T 15 --Dataset OD15_01.npz  --Model ConvLSTM  --LR 0.01
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 0 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City HZ --T 15 --Dataset OD15_01.npz  --Model CASCNN --LR 0.01 --CASCNN_C 1 --CASCNN_R 1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 0 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City HZ --T 15 --Dataset OD15_01.npz  --Model GEML  --LR 0.01 --GEML_GCN_Units 128,128  --GEML_LSTM_Units 128  --GEML_Weights 0.5,0.25,0.25
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 0 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City HZ --T 15 --Dataset OD15_01.npz  --Model AHGCSP --LR 0.01 --AHGCSP_GCN_Units 128  --AHGCSP_Dynamic_Units 32  --AHGCSP_T_Units 32  --AHGCSP_S_Units 32

     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 1 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SH --T 15 --Dataset OD15_04.npz  --Model HA  --LR 0.01
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 1 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SH --T 15 --Dataset OD15_04.npz  --Model ANN   --LR 0.01 --ANN_Units 128,128,1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 1 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SH --T 15 --Dataset OD15_04.npz  --Model GRU   --LR 0.01 --GRU_Units 128,128,1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 1 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SH --T 15 --Dataset OD15_04.npz  --Model GCN   --LR 0.01 --GCN_Units 128,128,1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 1 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SH --T 15 --Dataset OD15_04.npz  --Model ConvLSTM   --LR 0.01
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 1 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SH --T 15 --Dataset OD15_04.npz  --Model CASCNN --LR 0.01 --CASCNN_C 1 --CASCNN_R 1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 1 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SH --T 15 --Dataset OD15_04.npz  --Model GEML   --LR 0.01 --GEML_GCN_Units 128,128  --GEML_LSTM_Units 128  --GEML_Weights 0.5,0.25,0.25
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 1 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SH --T 15 --Dataset OD15_04.npz  --Model AHGCSP --LR 0.01 --AHGCSP_GCN_Units 128  --AHGCSP_Dynamic_Units 32  --AHGCSP_T_Units 32  --AHGCSP_S_Units 32

     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 3 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SZ --T 15 --Dataset OD15_05.npz  --Model HA --LR 0.01
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 3 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SZ --T 15 --Dataset OD15_05.npz  --Model ANN --LR 0.01 --ANN_Units 64,64,1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 3 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SZ --T 15 --Dataset OD15_05.npz  --Model GRU  --LR 0.01 --GRU_Units 64,64,1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 3 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SZ --T 15 --Dataset OD15_05.npz  --Model GCN  --LR 0.01 --GCN_Units 128,128,1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 3 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SZ --T 15 --Dataset OD15_05.npz  --Model ConvLSTM  --LR 0.001
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 3 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SZ --T 15 --Dataset OD15_05.npz  --Model CASCNN --LR 0.01 --CASCNN_C 1 --CASCNN_R 1
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 3 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SZ --T 15 --Dataset OD15_05.npz  --Model GEML --LR 0.01 --GEML_GCN_Units 128,128  --GEML_LSTM_Units 128  --GEML_Weights 0.5,0.25,0.25
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 3 --Note " "  --Repeat 1   --Batch 16  --Epochs 2 --City SZ --T 15 --Dataset OD15_05.npz  --Model AHGCSP --LR 0.01 --AHGCSP_GCN_Units 128  --AHGCSP_Dynamic_Units 32  --AHGCSP_T_Units 32  --AHGCSP_S_Units 32
)
pause

::cd ../bat
