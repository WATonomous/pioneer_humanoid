@echo off
py -3.11 "%~dp0keyboard_control.py"
if errorlevel 1 pause
