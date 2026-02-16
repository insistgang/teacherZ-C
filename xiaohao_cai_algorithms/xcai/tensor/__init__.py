"""
Tensor decomposition module containing Tucker, TT, and CUR decomposition methods.
"""

from xcai.tensor.tucker import TuckerDecomposer
from xcai.tensor.tt import TTDecomposer
from xcai.tensor.cur import CURDecomposer

__all__ = ['TuckerDecomposer', 'TTDecomposer', 'CURDecomposer']
