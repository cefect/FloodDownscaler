:: run pytest suite on all tests in the directory. and on the 

:: activate the environment
call "%~dp0../env/conda_activate.bat"
 
:: call pytest
ECHO starting tests in separate windows
python -m pytest --maxfail=10 %~dp0 
::%SRC_DIR%\fperf\tests %SRC_DIR%\coms\hp\tests

if errorlevel 1 exit /b 1
 

