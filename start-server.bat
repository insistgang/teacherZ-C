@echo off
echo ==========================================
echo  Xiaohao Cai 论文精读可视化系统
echo ==========================================
echo.
echo 正在启动本地服务器...
echo 访问地址: http://localhost:8080/web-viewer/
echo.
echo 按 Ctrl+C 停止服务器
echo.
cd /d D:\Documents\zx
python -m http.server 8080
pause
