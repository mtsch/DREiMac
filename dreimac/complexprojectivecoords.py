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
        standard_range=True,
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
            cohomdeath_rips, cohombirth_rips, cocycle = self.get_representative_cocycle(cohomology_class,homological_dimension)

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
        nu_ = lsqr(delta1, integer_cocycle_as_vector)[0]
        harmonic_representative = integer_cocycle_as_vector - delta1 @ nu_

        #turn cocycle into tensor
        def two_cocycle_to_tensor(cocycle:np.ndarray, dist_mat: np.ndarray, threshold: float, lookup_table: np.ndarray):
            n_points = dist_mat.shape[0]
            n_edges = (n_points * (n_points - 1)) // 2
            n_faces = number_of_simplices_of_dimension(2, n_points, lookup_table)

            res = np.zeros((n_points,n_points,n_points))

            #@jit(fastmath=True)
            def _get_res(
                cocycle:np.ndarray, 
                dist_mat: np.ndarray,
                threshold: float,
                lookup_table: np.ndarray,
                n_points: int,
                res: np.ndarray,
            ):
                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        if dist_mat[i, j] < threshold:
                            for k in range(j + 1, n_points):
                                if (
                                    dist_mat[i, k] < threshold
                                    and dist_mat[j, k] < threshold
                                ):
                                    flat_index = combinatorial_number_system_d2_forward(
                                        i, j, k, lookup_table
                                    )
                                    val = cocycle[flat_index]
                                    #012
                                    res[i,j,k] = val
                                    #021
                                    res[i,k,j] = -val
                                    #102
                                    res[j,i,k] = -val
                                    #210
                                    res[k,j,i] = -val
                                    #201
                                    res[k,i,j] = val
                                    #120
                                    res[j,k,i] = val

            _get_res(cocycle, dist_mat, threshold, lookup_table, n_points, res)
        
            return res

        def one_cocycle_to_tensor(cocycle:np.ndarray, dist_mat: np.ndarray, threshold: float, lookup_table: np.ndarray):
            n_points = dist_mat.shape[0]
            n_edges = (n_points * (n_points - 1)) // 2

            res = np.zeros((n_points,n_points))

            #@jit(fastmath=True)
            def _get_res(
                cocycle:np.ndarray, 
                dist_mat: np.ndarray,
                threshold: float,
                lookup_table: np.ndarray,
                n_points: int,
                res: np.ndarray,
            ):
                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        if dist_mat[i, j] < threshold:
                                flat_index = combinatorial_number_system_d1_forward(
                                    i, j, lookup_table
                                )
                                val = cocycle[flat_index]
                                res[i,j] = val
                                res[j,i] = -val

            _get_res(cocycle, dist_mat, threshold, lookup_table, n_points, res)
        
            return res

        nu = one_cocycle_to_tensor(nu_, self.dist_land_land_, rips_threshold, self.cns_lookup_table_)

        eta = two_cocycle_to_tensor(harmonic_representative,self.dist_land_land_, rips_threshold, self.cns_lookup_table_)

        class_map0 = np.zeros_like(varphi.T)

        n_data = self.X_.shape[0]
        for b in range(n_data):
            for i in range(self.n_landmarks_):
                class_map0[b,i] += nu[i, ball_indx[b]]
                for t in range(self.n_landmarks_):
                    class_map0[b,i] += varphi[t,b] * eta[i, ball_indx[b], t]

        class_map = np.exp( 2*np.pi*1j* class_map0 ) * np.sqrt(varphi.T)


        X = class_map.T
        # variance = np.zeros(X.shape[0])
        # dimension of projective space to project onto
        proj_dim = 1

        for i in range(class_map.shape[1]-proj_dim-1):
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
