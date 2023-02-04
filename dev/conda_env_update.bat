rem update the conda environment per the file

call activate.bat

echo on
call conda env update --file %ENV_FILE% --prune --name %ENV_NAME%

cmd.exe