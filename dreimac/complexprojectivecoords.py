import numpy as np
from numba import jit
import scipy.sparse as sparse
from scipy.sparse.linalg import lsqr
from scipy.optimize import LinearConstraint, milp, linprog
from .utils import PartUnity, CircleMapUtils, CohomologyUtils
from .emcoords import EMCoords
from .combinatorial import (
    combinatorial_number_system_table,
    combinatorial_number_system_forward,
    combinatorial_number_system_d1_forward,
    combinatorial_number_system_d2_forward,
    number_of_simplices_of_dimension,
)


class ComplexProjectiveCoords(EMCoords):
    def __init__(
        self, X, n_landmarks, distance_matrix=False, prime=41, maxdim=2, verbose=False
    ):
        """
        Parameters
        ----------
        X: ndarray(N, d)
            A point cloud with N points in d dimensions
        n_landmarks: int
            Number of landmarks to use
        distance_matrix: boolean
            If true, treat X as a distance matrix instead of a point cloud
        prime : int
            Field coefficient with which to compute rips on landmarks
        maxdim : int
            Maximum dimension of homology.  Only dimension 2 is needed for circular coordinates,
            but it may be of interest to see other dimensions (e.g. for a torus)
        """
        EMCoords.__init__(self, X, n_landmarks, distance_matrix, prime, maxdim, verbose)
        simplicial_complex_dimension = 2
        self.cns_lookup_table_ = combinatorial_number_system_table(
            n_landmarks, simplicial_complex_dimension
        )
        self.type_ = "complexprojective"

    def get_coordinates(
        self,
        perc=0.99,
        cohomology_class=0,
        partunity_fn=PartUnity.linear,
        check_and_fix_cocycle_condition=True,
    ):
        """

        TODO

        Parameters
        ----------
        perc : float
            Percent coverage
        inner_product : string
            Either 'uniform' or 'exponential'
        cohomology_class : integer
            TODO: explain
        partunity_fn: (dist_land_data, r_cover) -> phi
            A function from the distances of each landmark to a bump function

        Returns
        -------
        thetas: ndarray(n, N)
            TODO
        """

        HARDCODE = False

        if HARDCODE:
            # TODO: generalize
            data = self.X_
            landmarks = np.array(
                [
                    [0, 0, 1],
                    [2 * np.sqrt(2) / 3, 0, -1 / 3],
                    [-np.sqrt(2) / 3, np.sqrt(2 / 3), -1 / 3],
                    [-np.sqrt(2) / 3, -np.sqrt(2 / 3), -1 / 3],
                ]
            )
            n_landmarks = len(landmarks)
            self.n_landmarks_ = n_landmarks

            # NOTE: taking minimum between numbers and 1 because
            # when number is slighlty larger than 1 get nan with arccos
            dist_land_land = np.arccos(np.minimum(landmarks @ landmarks.T, 1))
            dist_land_data = np.arccos(np.minimum(landmarks @ data.T, 1))

            self.dist_land_data_ = dist_land_data
            self.dist_land_land_ = dist_land_land

            # get representative cocycle
            # TODO: generalize
            cohomdeath_rips, cohombirth_rips, cocycle = (
                1.92,
                4,
                np.array([[0, 1, 2, 1], [0, 1, 3, 0], [0, 2, 3, 0], [1, 2, 3, 0]]),
            )
        else:
            homological_dimension = 2
            cohomdeath_rips, cohombirth_rips, cocycle = self.get_representative_cocycle(
                cohomology_class, homological_dimension
            )

        standard_range = False

        ##### determine radius for balls
        r_cover, rips_threshold = EMCoords.get_cover_radius(
            self, perc, cohomdeath_rips, cohombirth_rips, standard_range
        )

        # compute partition of unity and choose a cover element for each data point
        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)

        # compute boundary matrix
        delta1 = CohomologyUtils.make_delta1(
            self.dist_land_land_, rips_threshold, self.cns_lookup_table_
        )

        # lift to integer cocycles
        integer_cocycle = CohomologyUtils.lift_to_integer_cocycle(
            cocycle, prime=self.prime_
        )
        integer_cocycle_as_vector = CohomologyUtils.sparse_cocycle_to_vector(
            integer_cocycle, self.cns_lookup_table_, self.n_landmarks_, int
        )

        # integrate cocycle
        integral = lsqr(delta1, integer_cocycle_as_vector)[0]
        harmonic_representative = integer_cocycle_as_vector - delta1 @ integral

        # assemble classifying map
        class_map0 = np.zeros_like(varphi.T)

        n_data = self.X_.shape[0]
        for b in range(n_data):
            for i in range(self.n_landmarks_):
                ordered_ij, sign_ordering_ij = CohomologyUtils.order_simplex(
                    np.array([i, ball_indx[b]])
                )
                index_ij = combinatorial_number_system_d1_forward(
                    ordered_ij[0], ordered_ij[1], self.cns_lookup_table_
                )
                if (
                    self.dist_land_land_[i, ball_indx[b]] < rips_threshold
                    and i != ball_indx[b]
                ):
                    class_map0[b, i] += sign_ordering_ij * integral[index_ij]

                for t in range(self.n_landmarks_):
                    ordered_ijt, sign_ordering_ijt = CohomologyUtils.order_simplex(
                        np.array([i, ball_indx[b],t])
                    )
                    index_ijt = combinatorial_number_system_d2_forward(
                        ordered_ijt[0], ordered_ijt[1], ordered_ijt[2], self.cns_lookup_table_
                    )
                    if (
                        self.dist_land_land_[i, ball_indx[b]] < rips_threshold
                        and self.dist_land_land_[i, t] < rips_threshold
                        and self.dist_land_land_[ball_indx[b], t] < rips_threshold
                        and i != ball_indx[b]
                        and i != t
                        and ball_indx[b] != t
                    ):
                        class_map0[b, i] += varphi[t, b] * sign_ordering_ijt * harmonic_representative[index_ijt]

        class_map = np.exp(2 * np.pi * 1j * class_map0) * np.sqrt(varphi.T)

        X = class_map.T
        # variance = np.zeros(X.shape[0])
        # dimension of projective space to project onto
        proj_dim = 1

        for i in range(class_map.shape[1] - proj_dim - 1):
            UU, S, _ = np.linalg.svd(X)
            # variance[-i] = np.mean(
            #    (np.pi/2 - np.arccos(np.abs(UU[:,-1].T @ X)))**2
            # )
            Y = np.conjugate(UU.T) @ X
            y = Y[-1, :]
            Y = Y[:-1, :]
            X = np.divide(Y, np.sqrt(1 - np.abs(y) ** 2))

        coords = np.zeros((X.shape[1], 2 * X.shape[0]))
        coords[:, ::2] = np.real(X).T
        coords[:, 1::2] = np.imag(X).T

        return coords
