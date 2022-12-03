REM build the conda environment from the file
call C:\Users\cefect\miniforge3\Scripts\activate.bat
conda env create -f %~dp0..\environment.yml --name fdsc
pause