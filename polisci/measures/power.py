import numpy as np


def __get_political_power_monte_carlo_winner(seats_normalized: np.ndarray) -> int:
    indicies = np.random.permutation(range(seats_normalized.shape[0]))
    cumulative_power = seats_normalized[indicies].cumsum()

    for i in range(len(indicies)):
        if cumulative_power[i] >= 0.5:
            return indicies[i]

    raise RuntimeError('We should never get here. Was seats_normalized actually normalized to sum to 1.0?')


def __get_political_power_monte_carlo_single_election(seats: np.ndarray, **kwargs) -> np.ndarray:
    assert len(seats.shape) == 1
    assert seats.shape[0] >= 2

    num_simulations: int = kwargs.get('num_simulations', 100)
    assert num_simulations >= 100

    # special case for number of seats == 2
    # no need for monte carlo here.
    if seats.shape[0] == 2:
        if seats[0] > seats[1]:
            return np.array([1.0, 0.0])
        elif seats[1] > seats[0]:
            return np.array([0.0, 1.0])
        else:
            return np.array([0.5, 0.5])

    # normalize seat proportions
    seats_normalized = seats / seats.sum()

    result = np.zeros(seats.shape, dtype=np.int)

    for _ in range(num_simulations):
        winner = __get_political_power_monte_carlo_winner(seats_normalized)
        result[winner] += 1

    # normalize result by number of wins
    return result * 1.0 / result.sum()


def get_political_power_monte_carlo(seats: np.ndarray, **kwargs) -> np.ndarray:
    if len(seats.shape) == 1:
        return __get_political_power_monte_carlo_single_election(seats, **kwargs)

    assert len(seats.shape) == 2

    num_elections = seats.shape[0]
    num_parties = seats.shape[1]

    power = np.zeros([num_elections, num_parties], dtype=np.float)

    for i in range(num_elections):
        election_seats = seats[i, :]
        election_power = __get_political_power_monte_carlo_single_election(election_seats, **kwargs)
        power[i, :] = election_power

    return power


