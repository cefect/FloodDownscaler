'''
Created on Jan. 6, 2023

@author: cefect

shared by all sessions
'''

from hp.oop import Session

class Master_Session(Session):
    def __init__(self, 
                 run_name='v1', #using v instead of r to avoid resolution confusion
                 **kwargs):
 
        super().__init__(run_name=run_name, **kwargs)