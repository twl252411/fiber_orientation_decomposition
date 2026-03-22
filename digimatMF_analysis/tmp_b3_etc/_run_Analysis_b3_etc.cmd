@echo off
setlocal
cd /d "D:\github\fiber_orientation_decomposition\digimatMF_analysis\tmp_b3_etc"
call "C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\DigimatMF_nogui.bat" input="D:\github\fiber_orientation_decomposition\digimatMF_analysis\tmp_b3_etc\Analysis_b3_etc.mat"
exit /b %errorlevel%
