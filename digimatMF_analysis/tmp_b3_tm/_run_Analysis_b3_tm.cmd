@echo off
setlocal
cd /d "D:\github\fiber_orientation_decomposition\digimatMF_analysis\tmp_b3_tm"
call "C:\MSC.Software\Digimat\2023.1\DigimatMF\exec\DigimatMF_nogui.bat" input="D:\github\fiber_orientation_decomposition\digimatMF_analysis\tmp_b3_tm\Analysis_b3_tm.mat"
exit /b %errorlevel%
