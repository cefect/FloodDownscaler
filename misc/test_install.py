'''
Created on Oct. 30, 2021

@author: cefect

testing installation
'''


#===============================================================================
# standard info
#===============================================================================
import os, sys

#system info
print('sys.version: '+sys.version)
print('exc_info(): ' + str(sys.exc_info()))
print('exe: %s'%sys.executable)



#systenm paths
print('system paths')
for k in sys.path: 
    print('    %s'%k)
    if not os.path.exists(k):
        print('WARNING: bad path: ' + k)
#print(sys.path)

#python paths
print('\npython path   ' )
for k in os.environ['PYTHONPATH'].split(';'):
    print('    ' + k)
    if not os.path.exists(k):
        print('WARNING: bad path: ' + k)

#directories
print('os.getcwd: %s'%os.getcwd())


#===============================================================================
# run test scripts
#===============================================================================
from hp.Q import Qproj

proj = Qproj()
proj._install_info()