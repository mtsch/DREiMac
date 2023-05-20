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

        ##homological_dimension = 2
        ##cohomdeath_rips, cohombirth_rips, cocycle = self.get_representative_cocycle(cohomology_class,homological_dimension)

        standard_range = False

        ##### determine radius for balls
        r_cover, rips_threshold = EMCoords.get_cover_radius(
            self, perc, cohomdeath_rips, cohombirth_rips, standard_range
        )

        # compute partition of unity and choose a cover element for each data point
        varphi, ball_indx = EMCoords.get_covering_partition(self, r_cover, partunity_fn)

        # compute boundary matrix
        # TODO: generalize
        ####delta0, edge_pair_to_row_index = CohomologyUtils.make_delta0(dist_land_land, threshold, self.cns_lookup_table_)
        delta1 = CohomologyUtils.make_delta1(
            dist_land_land, rips_threshold, self.cns_lookup_table_
        )

        # lift to integer cocycles
        integer_cocycle = CohomologyUtils.lift_to_integer_cocycle(
            cocycle, prime=self.prime_
        )
        integer_cocycle_as_vector = CohomologyUtils.sparse_cocycle_to_vector(
            integer_cocycle, self.cns_lookup_table_, n_landmarks, int
        )

        # integrate cocycle
        nu = lsqr(delta1, integer_cocycle_as_vector)[0]
        harmonic_representative = integer_cocycle_as_vector - delta1 @ nu

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


        eta = two_cocycle_to_tensor(harmonic_representative,dist_land_land, rips_threshold, self.cns_lookup_table_)

        class_map0 = np.zeros_like(varphi.T)

        n_data = self.X_.shape[0]
        for b in range(n_data):
            for i in range(n_landmarks):
                for t in range(n_landmarks):
                    class_map0[b,i] += varphi[t,b] * eta[i, ball_indx[b], t]

        class_map = np.exp( 2*np.pi*1j* class_map0 ) * np.sqrt(varphi.T)

        print(class_map.shape)


        #cocycle_matrix = np.ones((n_landmarks, n_landmarks))
        #cocycle_matrix[cocycle[:, 0], cocycle[:, 1]] = -1
        #cocycle_matrix[cocycle[:, 1], cocycle[:, 0]] = -1

        #class_map = np.sqrt(varphi.T)
        #for i in range(n_data):
        #    class_map[i, :] *= cocycle_matrix[ball_indx[i], :]
        ####N, d = data.shape
        ####s = landmarks.shape[0]
        ##### s = self.n_landmarks_

        ######## harcoded cover and partition of unity
        ####r = np.sort(dist_land_land, axis=1)[:, 1]
        ####import numpy.matlib

        ####U = dist_land_data < np.matlib.repmat(r, N, 1).T
        ######print("U", U)

        ####varphi = np.zeros((s, N))
        ####for j in range(0, s):
        ####    varphi[j, U[j, :]] = (r[j] - dist_land_data[j, U[j, :]]) ** 2
        ####sum_phi = np.sum(varphi, axis=0)
        ####varphi = varphi / sum_phi[np.newaxis, :]

        ####indx = np.zeros(N, dtype=int)

        ####for j in range(N):
        ####    indx[j] = np.argwhere(U[:, j])[0][0]

        ##### NOTE: the cover is not great
        ##### print("idx counts", sum(indx[indx==0].shape),sum(indx[indx==1].shape),sum(indx[indx==2].shape))
        ######print("indx", indx)
        ########

        ####h = np.zeros((s, s, N), dtype=complex)

        ####for j in range(s):
        ####    for k in range(s):
        ####        unordered_simplex = np.array([j, k], dtype=int)
        ####        ordered_simplex, sign = CohomologyUtils.order_simplex(unordered_simplex)
        ####        if ordered_simplex in edge_pair_to_row_index:
        ####            nu_val = sign * nu[edge_pair_to_row_index[ordered_simplex]]
        ####        else:
        ####            nu_val = 0

        ####        theta_average = 0
        ####        for l in range(s):
        ####            unordered_simplex = np.array([j, k, l], dtype=int)
        ####            ordered_simplex, sign = CohomologyUtils.order_simplex(
        ####                unordered_simplex
        ####            )
        ####            if ordered_simplex in simplex_to_vector_index:
        ####                theta_average += (
        ####                    sign
        ####                    * harmonic_representative[
        ####                        simplex_to_vector_index[ordered_simplex]
        ####                    ]
        ####                    * varphi[l]
        ####                )

        ####        h[j, k] = np.exp(2 * np.pi * 1j * (theta_average + nu_val))

        ####class_map = np.array(np.sqrt(varphi), dtype=complex)

        ####for j in range(N):
        ####    h_k_ind_j = h[:, indx[j]]
        ####    class_map[:, j] = class_map[:, j] * np.conjugate(h_k_ind_j[:, j])

        ##print("class map ", class_map)

        X = class_map.T
        # variance = np.zeros(X.shape[0])
        # dimension of projective space to project onto
        proj_dim = 1

        # for i in range(class_map.shape[0]-proj_dim-1):
        for i in [1, 2]:
            UU, S, _ = np.linalg.svd(X)
            ##print("singular vals", S)
            # print("norm UU", np.linalg.norm(UU))
            ##print("norm X", np.linalg.norm(X,axis=1))
            # variance[-i] = np.mean(
            #    (np.pi/2 - np.arccos(np.abs(UU[:,-1].T @ X)))**2
            # )
            Y = np.conjugate(UU.T) @ X
            y = Y[-1, :]
            ##print(np.linalg.norm(y))
            Y = Y[:-1, :]
            # print(Y.shape)
            ##print(np.sqrt( 1 - np.abs(y)**2 ).shape)
            X = np.divide(Y, np.sqrt(1 - np.abs(y) ** 2))
            ##print("X", X)

        print(X.shape)

        Z = np.zeros((2 * X.shape[0], X.shape[1]))
        print(Z.shape)
        Z[::2, :] = np.real(X)
        Z[1::2, :] = np.imag(X)
        projData = Z

        # XX = X
        # print(np.linalg.norm(XX))

        # for j in [1]:
        #    UU, _, _ = np.linalg.svd(XX)
        #    print("norm UU", np.linalg.norm(UU))
        #    #variance[-i] = np.mean(
        #    #    (np.pi/2 - np.arccos(np.abs(UU[:,-1].T @ X)))**2
        #    #)
        #    Y = UU.T @ XX
        #    #print(np.linalg.norm(Y,axis=1))
        #    y = Y[-1,:]
        #    #print(np.linalg.norm(y))
        #    Y = Y[:-1,:]
        #    XX = Y / np.sqrt( 1 - np.abs(y)**2 )

        return projData
