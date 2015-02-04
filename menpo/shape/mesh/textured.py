import numpy as np

from menpo.shape import PointCloud
from menpo.transform import Scale

from ..adjacency import mask_adjacency_array, reindex_adjacency_array
from .base import TriMesh


class TexturedTriMesh(TriMesh):
    r"""
    Combines a :map:`TriMesh` with a texture. Also encapsulates the texture
    coordinates required to render the texture on the mesh.

    Parameters
    ----------
    points : ``(n_points, n_dims)`` `ndarray`
        The array representing the points.
    tcoords : ``(N, 2)`` `ndarray`
        The texture coordinates for the mesh.
    texture : :map:`Image`
        The texture for the mesh.
    trilist : ``(M, 3)`` `ndarray` or `None`, optional
        The triangle list. If `None`, a Delaunay triangulation of
        the points will be used instead.
    copy: `bool`, optional
        If `False`, the points, trilist and texture will not be copied on
        assignment.
        In general this should only be used if you know what you are doing.
    """
    def __init__(self, points, tcoords, texture, trilist=None, copy=True):
        super(TexturedTriMesh, self).__init__(points, trilist=trilist,
                                              copy=copy)
        self.tcoords = PointCloud(tcoords, copy=copy)

        if not copy:
            self.texture = texture
        else:
            self.texture = texture.copy()

    def tcoords_pixel_scaled(self):
        r"""
        Returns a PointCloud that is modified to be suitable for directly
        indexing into the pixels of the texture (e.g. for manual mapping
        operations). The resulting tcoords behave just like image landmarks
        do:

        ::

            texture = texturedtrimesh.texture
            tc_ps = texturedtrimesh.tcoords_pixel_scaled()
            pixel_values_at_tcs = texture[tc_ps[: ,0], tc_ps[:, 1]]

        The operations that are performed are:

        - Flipping the origin from bottom-left to top-left
        - Scaling the tcoords by the image shape (denormalising them)
        - Permuting the axis so that

        Returns
        -------
        tcoords_scaled : :map:`PointCloud`
            A copy of the tcoords that behave like Image landmarks
        """
        scale = Scale(np.array(self.texture.shape)[::-1])
        tcoords = self.tcoords.points.copy()
        # flip the 'y' st 1 -> 0 and 0 -> 1, moving the axis to upper left
        tcoords[:, 1] = 1 - tcoords[:, 1]
        # apply the scale to get the units correct
        tcoords = scale.apply(tcoords)
        # flip axis 0 and axis 1 so indexing is as expected
        tcoords = tcoords[:, ::-1]
        return PointCloud(tcoords)

    def from_vector(self, flattened):
        r"""
        Builds a new :class:`TexturedTriMesh` given then `flattened` vector.
        Note that the trilist, texture, and tcoords will be drawn from self.

        Parameters
        ----------
        flattened : ``(N,)`` `ndarray`
            Vector representing a set of points.

        Returns
        --------
        trimesh : :map:`TriMesh`
            A new trimesh created from the vector with self's trilist.
        """
        return TexturedTriMesh(flattened.reshape([-1, self.n_dims]),
                               self.tcoords.points, self.texture,
                               trilist=self.trilist)

    def from_mask(self, mask):
        """
        A 1D boolean array with the same number of elements as the number of
        points in the TexturedTriMesh. This is then broadcast across the
        dimensions of the mesh and returns a new mesh containing only those
        points that were ``True`` in the mask.

        Parameters
        ----------
        mask : ``(n_points,)`` `ndarray`
            1D array of booleans

        Returns
        -------
        mesh : :map:`TexturedTriMesh`
            A new mesh that has been masked.
        """
        if mask.shape[0] != self.n_points:
            raise ValueError('Mask must be a 1D boolean array of the same '
                             'number of entries as points in this '
                             'TexturedTriMesh.')

        ttm = self.copy()
        if np.all(mask):  # Fast path for all true
            return ttm
        else:
            # Recalculate the mask to remove isolated vertices
            isolated_mask = self._isolated_mask(mask)
            # Recreate the adjacency array with the updated mask
            masked_adj = mask_adjacency_array(isolated_mask, self.trilist)
            ttm.trilist = reindex_adjacency_array(masked_adj)
            ttm.points = ttm.points[isolated_mask, :]
            ttm.tcoords.points = ttm.tcoords.points[isolated_mask, :]
            return ttm

    def _view_3d(self, figure_id=None, new_figure=False, textured=True,
                 **kwargs):
        r"""
        Visualize the :class:`TexturedTriMesh`. Only 3D objects are currently
        supported.

        Parameters
        ----------
        textured : `bool`, optional
            If ``True``, render the texture.

        Returns
        -------
        viewer : :class:`menpo.visualize.base.Renderer`
            The viewer object.

        Raises
        ------
        DimensionalityError
            If `self.n_dims != 3`.
        """
        if textured:
            try:
                from menpo3d.visualize import TexturedTriMeshViewer3d
                return TexturedTriMeshViewer3d(
                    figure_id, new_figure, self.points,
                    self.trilist, self.texture,
                    self.tcoords.points).render(**kwargs)
            except ImportError:
                from menpo.visualize import Menpo3dErrorMessage
                raise ImportError(Menpo3dErrorMessage)
        else:
            return super(TexturedTriMesh, self).view(figure_id=figure_id,
                                                     new_figure=new_figure,
                                                     **kwargs)

    def _view_2d(self, figure_id=None, new_figure=False, image_view=True,
                 render_lines=True, line_colour='r', line_style='-',
                 line_width=1., render_markers=True, marker_style='o',
                 marker_size=20, marker_face_colour='k', marker_edge_colour='k',
                 marker_edge_width=1., render_axes=True,
                 axes_font_name='sans-serif', axes_font_size=10,
                 axes_font_style='normal', axes_font_weight='normal',
                 axes_x_limits=None, axes_y_limits=None, figure_size=(10, 8),
                 label=None):
        import warnings
        warnings.warn(Warning('2D Viewing of Textured TriMeshes is not '
                              'supported, falling back to TriMesh viewing.'))
        return TriMesh._view_2d(
            self, figure_id=figure_id, new_figure=new_figure,
            image_view=image_view, render_lines=render_lines,
            line_colour=line_colour, line_style=line_style,
            line_width=line_width, render_markers=render_markers,
            marker_style=marker_style, marker_size=marker_size,
            marker_face_colour=marker_face_colour,
            marker_edge_colour=marker_edge_colour,
            marker_edge_width=marker_edge_width, render_axes=render_axes,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            axes_x_limits=axes_x_limits, axes_y_limits=axes_y_limits,
            figure_size=figure_size, label=label)

    def __str__(self):
        return '{}\ntexture_shape: {}, n_texture_channels: {}'.format(
            TriMesh.__str__(self), self.texture.shape, self.texture.n_channels)
