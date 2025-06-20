#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code was taken from https://github.com/JochenHinz/NAPDE_2024.
@author: Jochen Hinz
"""

import numpy as np
from functools import wraps

# shortcut for vectorisation.
_ = np.newaxis


def frozen(array: np.ndarray) -> np.ndarray:
  """
    Freeze a vector inplace and return it.

    Example
    -------

    >>> arr = np.zeros((10,), dtype=int)
    >>> print(arr[0])
        0
    >>> arr[0] = 1
    >>> print(arr[0])
        1
    >>> arr = np.zeros((10,), dtype=int)
    >>> arr = frozen(arr)
    >>> arr[0] = 1
        ERROR

    Both in and out of place will work.
    >>> arr = np.zeros((10,), dtype=int)
    >>> frozen(arr)
    >>> arr[0] = 1
        ERROR
  """
  array = np.asarray(array)
  array.flags.writeable = False
  return array


def freeze(fn):
  """
    Decorator that freezes the returned array inplace.

    Example
    -------

    def multiply(arr, val):
      return val * arr

    >>> arr = np.ones((5,), dtype=int)
    >>> new_arr = multiply(arr, 2)
    >>> print(new_arr)
        [2, 2, 2, 2, 2]
    >>> new_arr[0] = 10
    >>> print(new_arr)
        [10, 2, 2, 2, 2]

    @freeze
    def multiply(arr, val):
      return val * arr

    >>> arr = np.ones((5,), dtype=int)
    >>> new_arr = multiply(arr, 2)
    >>> print(new_arr)
        [2, 2, 2, 2, 2]
    >>> new_arr[0] = 10
        ERROR
  """
  @wraps(fn)
  def wrapper(*args, **kwargs):
    return frozen(fn(*args, **kwargs))
  return wrapper

class QuadRule:
  """
    Class representing a quadrature scheme.

    Attributes:
    -----------

    name : `str`
      The name of the quadrature scheme. For example '7 point Gauss of order 6'.
    order : `int`
      Order of the quadrature scheme.
    simplex_type : `str`
      Type of the simplex the scheme applies to. Must either be 'triangle'
      or 'line'.
      Other simplex types raise a NotImplementedError (for now).
    weights : `np.ndarray`
      array of weights of shape (nquadpoints,).
    points : `np.ndarray`
      array of quadrature points of shape (nquadpoints, 1) if simplex_type == 'line'
      or (nquadpoints, 2) if simplex_type == 'triangle'.
  """

  name: str
  order: int
  simplex_type: str
  weights: np.ndarray
  points: np.ndarray

  def __init__(self, name, order, simplex_type, weights, points):
    self.name = str(name)
    self.order = int(order)
    self.weights = frozen(weights)
    self.points = frozen(points)

    assert self.points.shape[1:] in ((1,), (2,))
    assert self.weights.shape == self.points.shape[:1]

    if simplex_type == 'line':
      assert self.ndims == 1
    elif simplex_type == 'triangle':
      assert self.ndims == 2
    else:
      raise NotImplementedError("Unknown simplex type '{}'.".format(simplex_type))

    self.simplex_type = str(simplex_type)

  @property
  def ndims(self):
    return self.points.shape[-1]


# Convenience function for instantiating the seven point gauss method of order 6
# for integrating over the reference triangle.
# >>> quadrule = seven_point_gauss_6()  # (don't forget parentheses)
seven_point_gauss_6 = lambda:  \
    QuadRule( name='7 point Gauss of order 6',
              order=6,
              simplex_type='triangle',
              weights=[ 9/80,
                        (155+np.sqrt(15))/2400,
                        (155+np.sqrt(15))/2400,
                        (155+np.sqrt(15))/2400,
                        (155-np.sqrt(15))/2400,
                        (155-np.sqrt(15))/2400,
                        (155-np.sqrt(15))/2400 ],
              points=[ [1/3, 1/3],
                       [(6+np.sqrt(15))/21, (6+np.sqrt(15))/21],
                       [(9-2*np.sqrt(15))/21, (6+np.sqrt(15))/21],
                       [(6+np.sqrt(15))/21, (9-2*np.sqrt(15))/21],
                       [(6-np.sqrt(15))/21, (9+2*np.sqrt(15))/21],
                       [(9+2*np.sqrt(15))/21, (6-np.sqrt(15))/21],
                       [(6-np.sqrt(15))/21, (6-np.sqrt(15))/21 ] ] )


def univariate_gauss(npoints=4):
  """ Gaussian quadrature scheme over the interval [0, 1]. """
  points, weights = np.polynomial.legendre.leggauss(npoints)
  # scale from interval [-1, 1] to interval [0, 1]
  points = (points + 1) / 2
  weights = .5 * weights
  return QuadRule(name='{npoint} point univariate Gaussian integration.',
                  order=2*npoints-1,
                  simplex_type='line',
                  weights=weights,
                  points=points[:, _])
