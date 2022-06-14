cd ../
for /L %%i in (1,1,1) do (
     C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe    main.py --GPU 0 --Note " "  --Repeat 5   --Batch 16  --Epochs 1000 --City HZ --T 15 --Dataset OD15_01.npz --P 8 --Model ANN   --LR 0.01 --ANN_Units 128,128,1
     )
pause

::cd ../bat
