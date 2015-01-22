from menpo.shape import PointCloud
from menpo.transform.groupalign.base import MultipleAlignment
from menpo.math import principal_component_decomposition as pca
import numpy as np
from scipy.spatial import cKDTree as KDTree
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp


class ICP(MultipleAlignment):
    def __init__(self, sources, target=None):
        self._test_iteration = []
        self.transformations = []
        self.point_correspondence = []

        # sort sources in number of points
        sources = np.array(sources)
        sortindex = np.argsort(np.array([s.n_points for s in sources]))[-1::-1]
        sort_sources = sources[sortindex]

        # Set first source as target (e.g. having most number of points)
        if target is None:
            target = sort_sources[0]

        super(ICP, self).__init__(sources, target)

        # Align Source with Target
        self.aligned_shapes = np.array(
            [self._align_source(s) for s in sources]
        )

    def _align_source(self, source, eps=1e-3, max_iter=100):

        # Initial Alignment using PCA
        # p0, r, sm, tm = self._pca_align(source)
        # transforms.append([r, sm, tm])
        p0 = source.points

        a_p, transforms, iters, point_corr = self._align(p0, eps, max_iter)
        iters = [source.points, p0] + iters

        self._test_iteration.append(iters)
        self.transformations.append(transforms)
        self.point_correspondence.append(point_corr)

        return PointCloud(a_p)

    def _align(self, i_s, eps, max_iter):
        # Align Shapes
        transforms = []
        iters = []
        it = 0
        pf = i_s
        n_p = i_s.shape[0]
        tolerance_old = tolerance = eps + 1
        while tolerance > eps and it < max_iter:
            pk = pf

            # Compute Closest Points
            yk, _ = self._cloest_points(pk)

            # Compute Registration
            pf, rot, smean, tmean = self._compute_registration(pk, yk)
            transforms.append([rot, smean, tmean])

            # Update source
            # pf = self._update_source(pk, np.hstack((qr, qt)))

            # Calculate Mean Square Matching Error
            tolerance_new = np.sum(np.power(pf - yk, 2)) / n_p
            tolerance = abs(tolerance_old - tolerance_new)
            tolerance_old = tolerance_new

            it += 1
            iters.append(pf)

        _, point_corr = self._cloest_points(pf)

        return pf, transforms, iters, point_corr

    def _pca_align(self, source):
        # Apply PCA on both source and target
        svecs, svals, smean = pca(source.points)
        tvecs, tvals, tmean = pca(self.target.points)

        # Compute Rotation
        svec = svecs[np.argmax(svals)]
        tvec = tvecs[np.argmax(tvals)]

        sang = np.arctan2(svec[1], svec[0])
        tang = np.arctan2(tvec[1], tvec[0])

        da = sang - tang

        tr = np.array([[np.cos(da), np.sin(da)],
                       [-1*np.sin(da), np.cos(da)]])

        # Compute Aligned Point
        pt = np.array([tr.dot(s - smean) + tmean for s in source.points])

        return pt, tr, smean, tmean

    def _update_source(self, p, q):
        return _apply_q(p, q)[:, :p.shape[1]]

    def _compute_registration(self, p, x):
        # Calculate Covariance
        up = np.mean(p, axis=0)
        ux = np.mean(x, axis=0)
        u = up[:, None].dot(ux[None, :])
        n_p = p.shape[0]
        cov = sum([pi[:, None].dot(xi[None, :])
                   for (pi, xi) in zip(p, x)]) / n_p - u

        # Apply SVD
        U, W, T = np.linalg.svd(cov)

        # Calculate Rotation Matrix
        qr = T.T.dot(U.T)
        # Calculate Translation Point
        pk = np.array([qr.dot(s - up) + ux for s in p])

        return pk, qr, up, ux

    def _cloest_points(self, source, target=None):
        points = np.array([self._closest_node(s, target) for s in source])

        return np.vstack(points[:, 0]), np.hstack(points[:, 1])

    def _closest_node(self, node, target=None):
        if target is None:
            target = self.target

        nodes = target
        if isinstance(target, PointCloud):
            nodes = np.array(target.points)

        dist_2 = np.sum((nodes - node) ** 2, axis=1)
        index = np.argmin(dist_2)
        return nodes[index], index


def nicp(source, target, eps=1e-3):
    r"""
    Deforms the source trimesh to align with to optimally the target.
    """
    n_dims = source.n_dims
    # Homogeneous dimension (1 extra for translation effects)
    h_dims = n_dims + 1
    points = source.points
    trilist = source.trilist

    # Configuration
    upper_stiffness = 101
    lower_stiffness = 1
    stiffness_step = 5
    transforms = []
    iters = []

    # Get a sorted list of edge pairs (note there will be many mirrored pairs
    # e.g. [4, 7] and [7, 4])
    edge_pairs = np.sort(np.vstack((trilist[:, [0, 1]],
                                    trilist[:, [0, 2]],
                                    trilist[:, [1, 2]])))

    # We want to remove duplicates - this is a little hairy, but basically we
    # get a view on the array where each pair is considered by numpy to be
    # one item
    edge_pair_view = np.ascontiguousarray(edge_pairs).view(
        np.dtype((np.void, edge_pairs.dtype.itemsize * edge_pairs.shape[1])))
    # Now we can use this view to ask for only unique edges...
    unique_edge_index = np.unique(edge_pair_view, return_index=True)[1]
    # And use that to filter our original list down
    unique_edge_pairs = edge_pairs[unique_edge_index]

    # record the number of unique edges and the number of points
    n = points.shape[0]
    m = unique_edge_pairs.shape[0]

    # Generate a "node-arc" (i.e. vertex-edge) incidence matrix.
    M = np.zeros((m, n))
    M[range(m), unique_edge_pairs[:, 0]] = -1
    M[range(m), unique_edge_pairs[:, 1]] = 1

    # weight matrix
    G = np.identity(n_dims + 1)

    # build the kD-tree
    print('building KD-tree for target...')
    kdtree = KDTree(target.points)

    # init transformation
    X_prev = np.zeros((n_dims, n_dims + 1))
    X_prev = np.tile(X_prev, n).T
    v_i = points

    # start nicp
    # for each stiffness
    stiffness = range(upper_stiffness, lower_stiffness, -stiffness_step)
    M_kron_G = sp.kron(M, G)
    errs = []

    for alpha in stiffness:
        print(alpha)
        # get the term for stiffness
        alpha_M_kron_G = alpha * M_kron_G

        # iterate until X converge
        while True:
            # find nearest neighbour
            match = kdtree.query(v_i)[1]

            # formulate target and template data, and distance term
            U = target.points[match, :]

            D = np.zeros((n, n * h_dims))
            for k in range(n):
                D[k, k * h_dims: k * h_dims + n_dims] = v_i[k, :]
                D[k, k * h_dims + n_dims] = 1

            # correspondence detection for setting weight
            # add distance term
            A_s = sp.vstack((alpha_M_kron_G, D))
            B_s = sp.vstack((np.zeros((alpha_M_kron_G.shape[0], n_dims)), U))
            X_s = spsolve(A_s.T.dot(A_s), A_s.T.dot(B_s))
            X = X_s.todense()

            # deform template
            v_i = D.dot(X)
            err = np.linalg.norm(X_prev - X, ord='fro')
            errs.append([alpha, err])
            X_prev = X

            transforms.append(X)
            iters.append(v_i)

            if err / np.sqrt(np.size(X_prev)) < eps:
                break

    # final result
    fit = v_i
    _, point_corr = kdtree.query(fit)
    return fit, transforms, iters, point_corr
