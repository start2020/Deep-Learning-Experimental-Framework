cd ../
for /L %%i in (1,1,1) do (
C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe inputs.py --P 8 --City HZ --T 15 --Dataset OD15_01.npz
::C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe inputs.py --P 8 --City SH --T 15 --Dataset OD15_04.npz
C:\Users\ASUS\AppData\Local\Programs\Python\Python37\python.exe inputs.py --P 8 --City SZ --T 15 --Dataset OD15_05.npz
)
pause

::cd ../bat
