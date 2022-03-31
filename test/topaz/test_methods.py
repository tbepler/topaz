import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from topaz.methods import GE_KL, PN, PU, GE_binomial, autoencoder_loss


def test_autoencoder_loss():
    pass


class TestPN():
    def test_pn_init(self):
        pass

    def test_pn_step(self):
        pass


def TestPU():
    def test_pu_init(self):
        pass

    def test_pu_step(self):
        pass


def TestGE_KL():
    def test_gekl_init(self):
        pass

    def test_gekl_step(self):
        pass


def TestGE_binomial():
    def test_gebinomial_init(self):
        pass

    def test_gebinomial_step(self):
        pass
