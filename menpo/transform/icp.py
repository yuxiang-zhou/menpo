from menpo.shape import PointCloud
from menpo.transform.groupalign.base import MultipleAlignment
from menpo.math import principal_component_decomposition as pca
from menpo.shape import TriMesh
import numpy as np
from scipy.spatial import KDTree
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
    n_dims = source.n_dims
    source_points = source.points

    # Configuration
    higher = 101
    lower = 1
    step = 5
    transforms = []
    iters = []

    # Build TriMesh Source
    tplt_tri = TriMesh(source_points).trilist

    # Generate Edge List
    tplt_edge = tplt_tri[:, [0, 1]]
    tplt_edge = np.vstack((tplt_edge, tplt_tri[:, [0, 2]]))
    tplt_edge = np.vstack((tplt_edge, tplt_tri[:, [1, 2]]))
    tplt_edge = np.sort(tplt_edge)

    # Get Unique Edge List
    b = np.ascontiguousarray(tplt_edge).view(
        np.dtype((np.void, tplt_edge.dtype.itemsize * tplt_edge.shape[1]))
    )
    _, idx = np.unique(b, return_index=True)
    tplt_edge = tplt_edge[idx]

    # init
    m = tplt_edge.shape[0]
    n = source_points.shape[0]

    # get node-arc incidence matrix
    M = np.zeros((m, n))
    M[range(m), tplt_edge[:, 0]] = -1
    M[range(m), tplt_edge[:, 1]] = 1

    # weight matrix
    G = np.identity(n_dims + 1)

    # build the kD-tree
    target_2d = target.points
    kdOBJ = KDTree(target_2d)

    # init transformation
    prev_X = np.zeros((n_dims, n_dims + 1))
    prev_X = np.tile(prev_X, n).T
    tplt_i = source_points

    # start nicp
    # for each stiffness
    sf = range(higher, lower, -step)
    sf_kron = sp.kron(M, G)
    errs = []

    for alpha in sf:
        # get the term for stiffness
        sf_term = alpha*sf_kron
        # iterate until X converge
        while True:
            # find nearest neighbour
            _, match = kdOBJ.query(tplt_i)

            # formulate target and template data, and distance term
            U = target_2d[match, :]

            point_size = n_dims + 1
            D = np.zeros((n, n*point_size))
            for k in range(n):
                D[k, k * point_size: k * point_size + n_dims] = tplt_i[k, :]
                D[k, k * point_size + n_dims] = 1

            # % correspondence detection for setting weight
            # add distance term
            sA = sp.vstack((sf_term, D))
            sB = sp.vstack((np.zeros((sf_term.shape[0], n_dims)), U))
            sX = sp.linalg.spsolve(sA.T.dot(sA), sA.T.dot(sB))
            sX = sX.todense()

            # deform template
            tplt_i = D.dot(sX)
            err = np.linalg.norm(prev_X-sX, ord='fro')
            errs.append([alpha, err])
            prev_X = sX

            transforms.append(sX)
            iters.append(tplt_i)

            if err/np.sqrt(np.size(prev_X)) < eps:
                break

    # final result
    fit_2d = tplt_i
    _, point_corr = kdOBJ.query(fit_2d)
    return fit_2d, transforms, iters, point_corr
