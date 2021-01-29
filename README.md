# polisci
Political Science in Python

This project will mostly be about reimplementing my [undergraduate dissertation](https://www.dropbox.com/s/grwyfydi6btn0ht/CS380_Michael_Fowlie_Final.pdf?dl=0) in Python with a re-usable library, which can then serve as a basis of further research by myself or others.

## Abstract

It is claimed in (Carey and Hix, 2011) that the trade-off between expected fairness and stability in
election systems is non-linear and that small medium sized District Magnitudes are optimal, based
on outcomes of elections in the real world. Does this apply in general or can this effect be explained
by other factors? We test the hypothesis that the trade-off is non-linear against artificial societies
and across a range of voting systems. We use the Spatial Model, Polya Eggenberger Model and
a preference swapping model and the STV, District Proportional and SM rules. Our results show
clearly that there is non-linearity in this trade-off however the shape of the curve differs greatly
depending on the system we use. We compare these different outcomes under a range of different
undesirability functions that value the properties differently. We find that a District Magnitude
of DM = 3, DM = 8 or DM = 20 is optimal, depending on how the properties are valued with
respect to each other.

## Political Power
Represents the percentage of time that a particular party has the deciding vote. In reality there are issues with this model since it doesn't consider how some parties will tend to vote the same way on many issues.

This is an expensive computation to compute as it has O(n!) complexity to evaluate correctly. We can approximate it using monte carlo methods.

As an alternative, a Tensor Flow model has been trained to approximate this function.

## Measures

Unfairness and instability are both loss functions for elections. We want to minimize each. Unfortunately it is difficult to minimize both at the same time.

### Unfairness measures

- Loosemorehanby index
- Gallagher index

### Instability measures

- Effective number of parties (seats, or power)

## Societies

We define models for societies whereby we can compute batches of votes with preference and weight data.

### Spatial Model

Defines parties, electorates, and voters to exist in n dimensional space. Voters in a given electorate are centered around the electorate centroid. The standard deviations of each can be varied on a per dimension basis. Voters prefer parties that are closer to them by Minkowski distance.

This could be extended to have parties represented by a cluster of points rather than a single point. The distance between a voter and a party would be the minimum distance to any of the centroids for that party. The electorates could be represented by a cluster of points.
