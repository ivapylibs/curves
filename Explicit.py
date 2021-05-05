import numpy as np
from scipy.linalg import pascal
from matplotlib import pyplot as plt
from CurveBase import CurveBase
import pdb


class Explicit(CurveBase):
    def __init__(self, curveFunc, tspan=[0,1]):
        super().__init__(tspan)
        self.curveFunc = curveFunc
    
    def segment(self, tspan):
        return Explicit(self.curveFunc, tspan)
    
    def x(self, t):
        return self.curveFunc(t)
