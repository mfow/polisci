import numpy as np


def __normalize(data: np.ndarray) -> np.ndarray:
    totals = data.sum(axis=-1)
    totals = np.expand_dims(totals, axis=-1)
    return data.astype('float') / totals


def __unfairness_common(votes, seats):
    assert isinstance(votes, np.ndarray)
    assert isinstance(seats, np.ndarray)
    assert votes.shape == seats.shape
    votes = __normalize(votes)
    seats = __normalize(seats)
    return votes, seats


def loosemorehanby_index(votes: np.ndarray, seats: np.ndarray) -> np.ndarray:
    votes, seats = __unfairness_common(votes, seats)
    deltas = votes - seats
    return np.abs(deltas).sum(axis=-1) / 2.0


def gallagher_index(votes: np.ndarray, seats: np.ndarray) -> np.ndarray:
    votes, seats = __unfairness_common(votes, seats)
    deltas = votes - seats
    return np.sqrt((deltas * deltas).sum(axis=-1) * 0.5)
