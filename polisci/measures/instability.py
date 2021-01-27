import numpy as np
from .power import get_political_power_monte_carlo


def effective_num_of_parties(seats: np.ndarray) -> np.ndarray:
    seat_proportions = seats.astype('float') / np.expand_dims(np.sum(seats, axis=-1), axis=-1)
    return 1.0 / np.sum(seat_proportions * seat_proportions, axis=-1)


def effective_num_of_parties_power(seats: np.ndarray, **kwargs) -> float:
    power = get_political_power_monte_carlo(seats, **kwargs)
    return effective_num_of_parties(power)

