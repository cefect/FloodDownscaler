#print python info
import sys, os
print(sys.version)
print(sys.executable)

#print system info
import platform
print(platform.system(), platform.release())

#check definitions.py
from definitions import src_dir, src_name, logcfg_file, wrk_dir

assert os.path.exists(src_dir), src_dir
print(f'src_dir: {src_dir}')

print(f'src_name: {src_name}')

assert os.path.exists(logcfg_file), logcfg_file
print(f'logcfg_file: {logcfg_file}')

assert os.path.exists(wrk_dir), wrk_dir
print(f'wrk_dir:{wrk_dir}')

