import numpy as np
from scipy.linalg import pascal
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb
import abc

class CurveBase(abc.ABC):

    def __init__(self, tspan=[]):
        """Initializes the curve with a valid time period

        Keyword arguments:
        tspan -- a list containing the starting and ending times (default value [])
        """
        self.tspan = tspan

    def setDomain(self, tspan):
        self.tspan = tspan

    def isValid(self, t):
        return t <= self.tspan[1] and t >= self.tspan[0]

    @abc.abstractmethod 
    def x(self, t):
        """Evaluates the curve at points (t)

        Keyword arguments:
        t -- Vector containing evaluation times
        """
        return