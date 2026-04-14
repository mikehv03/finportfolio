"""
finportfolio
------------
A portfolio theory and asset pricing library implementing:
- Markowitz mean-variance optimization
- CAPM, APT, and the Gordon Growth Model
- Single Index and Fama-French factor models
- Portfolio performance and risk metrics
- Market data loading via yfinance
"""

from . import data
from . import returns
from . import equilibrium
from . import factors
from . import optimization
from . import performance

__all__ = [
    "data",
    "returns",
    "equilibrium",
    "factors",
    "optimization",
    "performance",
]

__version__ = "0.2.0"
__author__ = "Miguel Herrera"