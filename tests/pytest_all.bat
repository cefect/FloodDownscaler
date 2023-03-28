:: run pytest suite on all tests in the directory. and on the 

:: activate the environment
call "%~dp0../env/conda_activate.bat"
 
:: call pytest
ECHO starting tests in separate windows
start cmd.exe /k python -m pytest --maxfail=10 %~dp0  
start cmd.exe /k python -m pytest --maxfail=10 %SRC_DIR%\fperf\tests
start cmd.exe /k python -m pytest --maxfail=10 %SRC_DIR%\coms\hp\tests

cmd.exe /k