import numpy as np
from .society import Society, ElectorateVoteBatch
from typing import Dict, Iterable
from ..distances import get_distances


class SpatialModel(Society):
    def __init__(self, **kwargs):
        super(SpatialModel, self).__init__(**kwargs)
        self.ndim: int = kwargs.get('ndim', 1)
        assert isinstance(self.ndim, int)
        assert self.ndim >= 1

        self.voters_per_electorate: int = kwargs.get('voters_per_electorate', 1000)
        self.batch_size: int = kwargs.get('batch_size', 256)
        self.distance: str = kwargs.get('distance', 'l2')

        self.scales: Dict[str, np.ndarray] = kwargs.get('scales', {})

        def get_scale(name):
            scale_name = 'scale_{}'.format(name)

            if name in self.scales:
                assert scale_name not in kwargs
                return

            scale: np.ndarray = kwargs.get(scale_name)

            if scale is None:
                scale = np.ones([self.ndim], dtype=np.float)
            elif isinstance(scale, list):
                scale = np.array(scale)

            assert isinstance(scale, np.ndarray)
            assert len(scale.shape) == 1
            assert scale.shape[0] == self.ndim
            assert scale.dtype == np.float

            self.scales[name] = scale

        get_scale('party')
        get_scale('electorate')
        get_scale('voter')

    @property
    def party_scale(self) -> np.ndarray:
        return self.scales['party']

    @property
    def electorate_scale(self) -> np.ndarray:
        return self.scales['electorate']

    @property
    def voter_scale(self) -> np.ndarray:
        return self.scales['voter']

    def simulate(self, **kwargs) -> Iterable[ElectorateVoteBatch]:
        num_elections: int = kwargs.get('num_elections', 1)

        for election_id in range(num_elections):
            party_offsets = (np.random.randn(self.num_parties, self.ndim) *
                             np.expand_dims(self.party_scale, axis=0))

            for electorate_id in range(self.num_electorates):
                electorate_offsets = np.random.randn(self.ndim) * self.electorate_scale

                remaining_votes = self.voters_per_electorate

                while remaining_votes > 0:
                    current_batch_size = min(remaining_votes, self.batch_size)

                    voter_positions = (np.random.randn(current_batch_size, self.ndim) *
                                       np.expand_dims(self.voter_scale, axis=0)) + electorate_offsets

                    distances = get_distances(party_offsets, voter_positions, self.distance)
                    preferences = np.argsort(distances, axis=0)
                    preferences = np.swapaxes(preferences, 0, 1)

                    yield ElectorateVoteBatch(election_id=election_id,
                                              electorate_id=electorate_id,
                                              vote_batch=preferences)

                    remaining_votes -= current_batch_size
