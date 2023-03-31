import platform
from hp.subproc import sub_cmd


def test_sub_cmd():
    """simple command tester for Popen on linux"""
    #get system command
    print(platform.system(), platform.release())
    if 'Linux' in platform.system():
        cmd = 'uname -a'
        kwargs = dict(shell=True)
    else:
        raise NotImplementedError('dome')
    sub_cmd(cmd, **kwargs)
