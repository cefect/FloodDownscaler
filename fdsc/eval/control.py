'''
Created on Mar. 27, 2023

@author: cefect

running evaluation on downscaling results
'''

from fdsc.base import DscBaseSession
from fperf.pipeline import ValidateSession

class Dsc_Eval_Session(DscBaseSession, ValidateSession):
    pass