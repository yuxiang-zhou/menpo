.. _api-transform-index:

:mod:`menpo.transform`
======================

Homogeneous Transforms
----------------------

.. toctree::
  :maxdepth: 1

  Homogeneous
  Affine
  Similarity
  Rotation
  Translation
  Scale
  UniformScale
  NonUniformScale


Alignments
----------

.. toctree::
  :maxdepth: 1

  ThinPlateSplines
  PiecewiseAffine
  AlignmentAffine
  AlignmentSimilarity
  AlignmentRotation
  AlignmentTranslation
  AlignmentUniformScale


Group Alignments
----------------

.. toctree::
  :maxdepth: 1

  GeneralizedProcrustesAnalysis


Composite Transforms
--------------------

.. toctree::
  :maxdepth: 1

  TransformChain


Radial Basis Functions
----------------------

.. toctree::
  :maxdepth: 1

  R2LogR2RBF
  R2LogRRBF


Abstract Bases
--------------

.. toctree::
  :maxdepth: 1

  Transform
  Transformable
  ComposableTransform
  Invertible
  Alignment
  MultipleAlignment
  DiscreteAffine

Performance Specializations
---------------------------

Mix-ins that provide fast vectorized varients of methods.

.. toctree::
  :maxdepth: 1

  VComposable
  VInvertible
