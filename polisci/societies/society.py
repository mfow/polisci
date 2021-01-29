from abc import ABC, abstractmethod
import numpy as np


class Society(ABC):
    def __init__(self, **kwargs):
        self.num_parties: int = kwargs.get('num_parties')
        self.num_electorates: int = kwargs.get('num_electorates')

        assert isinstance(self.num_parties, int)
        assert isinstance(self.num_electorates, int)
        assert self.num_parties >= 2
        assert self.num_electorates >= 1

    @abstractmethod
    def simulate(self, **kwargs) -> np.ndarray:
        raise NotImplementedError
