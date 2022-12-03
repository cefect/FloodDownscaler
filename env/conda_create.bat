REM build the conda environment from the file
call setup_conda
conda env create -f %ENV_FILE% --name %ENV_NAME%
pause