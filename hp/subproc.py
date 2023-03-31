"""
subprocess helpers

"""
import subprocess


def sub_cmd(cmd, **kwargs):
    """execute a command as a subprocess"""
    print(f'with cmd={cmd}')
    err_str = ''
    with subprocess.Popen(cmd,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          text=True, bufsize=1,
                          **kwargs) as proc:

        for line in proc.stdout:
            if line in ['', '\n']: continue  # skip empty lines
            # print(f'    {line}', end='') #print stdout
            print('    %s' % line.replace('\n', ''))  # log stdout

        for line in proc.stderr:
            print(line.replace('\n', ''))
            err_str += line

            # check the error status
    msg = f'Popen got {proc.returncode} for \n    {proc.args}'
    if (proc.returncode != 0) or (err_str != ''):
        raise RuntimeError(msg + '\n' + err_str)

    print(msg)

    return proc