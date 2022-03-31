from re import L
import numpy as np
from topaz.model.features.basic import BasicConv, Conv63, Conv127


class TestBasicConv():

    def test_init(self):
        layers = [5]
        units = 5
        model = BasicConv(layers, units)

    
    def test_fill(self):
        pass


    def test_unfill(self):
        pass


    def test_forward(self):
        pass



class TestConv127():
    def test_conv127(self):
        pass



class TestConv63():
    def test_conv63(self):
        pass  