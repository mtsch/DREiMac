import numpy as np
import time
from .utils import PartUnity, EquivariantPCA
from .emcoords import EMCoords


class ProjectiveCoords(EMCoords):
    """
    Object that performs multiscale real projective coordinates via
    persistent cohomology of sparse filtrations (Jose Perea 2018).

    Parameters
    ----------
    X: ndarray(N, d)
        A point cloud with N points in d dimensions
    n_landmarks: int
        Number of landmarks to use
    distance_matrix: boolean
        If true, treat X as a distance matrix instead of a point cloud
    maxdim : int
        Maximum dimension of homology. Only dimension 1 is needed for circular coordinates,
        but it may be of interest to see other dimensions (e.g. for a torus)
    partunity_fn: ndarray(n_landmarks, N) -> ndarray(n_landmarks, N)
        A partition of unity function

    """

    def __init__(self, X, n_landmarks, distance_matrix=False, maxdim=1, verbose=False):
        EMCoords.__init__(
            self,
            X=X,
            n_landmarks=n_landmarks,
            distance_matrix=distance_matrix,
            prime=2,
            maxdim=maxdim,
            verbose=verbose,
        )
        self.type_ = "proj"
        # GUI variables
        self.selected = set([])
        self.u = np.array([0, 0, 1])

    def get_coordinates(
        self,
        perc=0.9,
        cocycle_idx=0,
        proj_dim=2,
        partunity_fn=PartUnity.linear,
        standard_range=True,
    ):
        """
        Get real projective coordinates.

        Parameters
        ----------
        perc : float
            Percent coverage. Must be between 0 and 1.
        cocycle_idx : list
            Add the cocycles together, sorted from most to least persistent
        proj_dim : integer
            Dimension down to which to project the data
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function
        standard_range : bool
            Whether to use the parameter perc to choose a filtration parameter that guarantees
            that the selected cohomology class represents a class in the Cech complex.

        Returns
        -------
        {'variance': ndarray(N-1)
            The variance captured by each dimension
        'X': ndarray(N, proj_dim+1)
            The projective coordinates
        }

        """

        n_landmarks = self.n_landmarks_
        n_data = self.X_.shape[0]

        homological_dimension = 1
        cohomdeath_rips, cohombirth_rips, cocycle = self.get_representative_cocycle(
            cocycle_idx, homological_dimension
        )

        r_cover, _ = EMCoords.get_cover_radius(
            self, perc, cohomdeath_rips, cohombirth_rips, standard_range
        )

        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)

        cocycle_matrix = np.ones((n_landmarks, n_landmarks))
        cocycle_matrix[cocycle[:, 0], cocycle[:, 1]] = -1
        cocycle_matrix[cocycle[:, 1], cocycle[:, 0]] = -1
        class_map = np.sqrt(varphi.T)
        for i in range(n_data):
            class_map[i, :] *= cocycle_matrix[ball_indx[i], :]

        epca = EquivariantPCA.ppca(class_map, proj_dim, self.verbose)
        self.variance_ = epca["variance"]

        return epca["X"]
