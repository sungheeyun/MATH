from abc import ABC, abstractmethod

from numpy import ndarray


class AlgorithmBase(ABC):

    @abstractmethod
    def optimize(
            self,
            initial_point: ndarray,
    ):
        pass


