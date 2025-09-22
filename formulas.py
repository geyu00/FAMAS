import math
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class SuspiciousnessFormula(ABC):
    @abstractmethod
    def compute(self, ncf: float, ncs: float, Nf: float, Ns: float, weight=1.0, mpIDF=1.0) -> float:
        pass


class Ochiai(SuspiciousnessFormula):
    def compute(self, ncf: float, ncs: float, Nf: float, Ns: float, weight=1.0, mpIDF=1.0) -> float:
        if ncf + ncs == 0 or Nf == 0:
            return 0.0
        return weight*ncf / (((weight*ncf + (Nf - ncf)) * (weight*ncf + mpIDF*ncs)*1.0) ** 0.5)


class Tarantula(SuspiciousnessFormula):
    def compute(self, ncf: float, ncs: float, Nf: float, Ns: float, weight=1.0, mpIDF=1.0) -> float:
        if Nf == 0 or Ns == 0:
            return 0.0
        fail_ratio = (weight *ncf*1.0) / (1.0*(Nf-ncf)+weight*ncf)
        pass_ratio = (mpIDF*ncs*1.0) / (1.0*(mpIDF*ncs+(Ns-ncs)))
        if fail_ratio + pass_ratio == 0:
            return 0.0
        return fail_ratio / (fail_ratio + pass_ratio)


class Jaccard(SuspiciousnessFormula):
    def compute(self, ncf: float, ncs: float, Nf: float, Ns: float, weight=1.0, mpIDF=1.0) -> float:
        if ncf + ncs + (Nf - ncf) == 0:
            return 0.0

        return weight*ncf / (1.0*(weight *ncf + mpIDF*ncs + (Nf - ncf)))





class DStar2(SuspiciousnessFormula):
    def compute(self, ncf: float, ncs: float, Nf: float, Ns: float, weight=1.0, mpIDF=1.0) -> float:
        denom = mpIDF*ncs + (Nf - ncf)  # passed(e) + not_exec_failed(e)
        if denom == 0:
            return float('inf')  
        return ((1.0*weight *ncf) ** 2) / (denom*1.0)


class Kulczynski2(SuspiciousnessFormula):
    def compute(self, ncf: float, ncs: float, Nf: float, Ns: float, weight=1.0, mpIDF=1.0) -> float:
        denom1 = weight *ncf + (Nf - ncf)  
        denom2 = weight *ncf + mpIDF*ncs
        if denom1 == 0 or denom2 == 0:
            return 0.0
        return  0.5 *((weight *ncf) / ( 1.0 * denom1) + (weight *ncf * 1.0) / ( 1.0 * denom2))

class Ochiai_Kulczynski2(SuspiciousnessFormula):
    def compute(self, ncf: float, ncs: float, Nf: float, Ns: float, weight=1.0, mpIDF=1.0) -> float:
        if ncf + ncs == 0 or Nf == 0:
            return 0.0
        s1 = weight*ncf / (((weight*ncf + (Nf - ncf)) * (weight*ncf + mpIDF*ncs)*1.0) ** 0.5)
        
        denom1 = weight *ncf + (Nf - ncf)  
        denom2 = weight *ncf + mpIDF*ncs
        if denom1 == 0 or denom2 == 0:
            return 0.0
        s2 = 0.5 *((weight *ncf) / ( 1.0 * denom1) + (weight *ncf * 1.0) / ( 1.0 * denom2))
        
        return (2*s1 + 2*s2)/4



FORMULA_MAP = {
    "ochiai": Ochiai(),
    "tarantula": Tarantula(),
    "jaccard": Jaccard(),
    "dstar2": DStar2(),
    "kulczynski2": Kulczynski2(),
    # "ochiai_kulczynski2": Ochiai_Kulczynski2()
}

def get_formula(name: str) -> SuspiciousnessFormula:
    name = name.lower()
    if name not in FORMULA_MAP:
        raise ValueError(f"Unsupported formula: {name}. Choose from {list(FORMULA_MAP.keys())}")
    return FORMULA_MAP[name]
