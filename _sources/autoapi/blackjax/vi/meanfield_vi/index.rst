:py:mod:`blackjax.vi.meanfield_vi`
==================================

.. py:module:: blackjax.vi.meanfield_vi


Module Contents
---------------

Classes
~~~~~~~

.. autoapisummary::

   blackjax.vi.meanfield_vi.MFVIState
   blackjax.vi.meanfield_vi.MFVIInfo
   blackjax.vi.meanfield_vi.meanfield_vi



Functions
~~~~~~~~~

.. autoapisummary::

   blackjax.vi.meanfield_vi.step
   blackjax.vi.meanfield_vi.sample
   blackjax.vi.meanfield_vi.generate_meanfield_logdensity



.. py:class:: MFVIState




   .. py:attribute:: mu
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: rho
      :type: blackjax.types.ArrayTree

      

   .. py:attribute:: opt_state
      :type: optax.OptState

      


.. py:class:: MFVIInfo




   .. py:attribute:: elbo
      :type: float

      


.. py:function:: step(rng_key: blackjax.types.PRNGKey, state: MFVIState, logdensity_fn: Callable, optimizer: optax.GradientTransformation, num_samples: int = 5, stl_estimator: bool = True) -> tuple[MFVIState, MFVIInfo]

   Approximate the target density using the mean-field approximation.

   :param rng_key: Key for JAX's pseudo-random number generator.
   :param init_state: Initial state of the mean-field approximation.
   :param logdensity_fn: Function that represents the target log-density to approximate.
   :param optimizer: Optax `GradientTransformation` to be used for optimization.
   :param num_samples: The number of samples that are taken from the approximation
                       at each step to compute the Kullback-Leibler divergence between
                       the approximation and the target log-density.
   :param stl_estimator: Whether to use stick-the-landing (STL) gradient estimator :cite:p:`roeder2017sticking` for gradient estimation.
                         The STL estimator has lower gradient variance by removing the score function term
                         from the gradient. It is suggested by :cite:p:`agrawal2020advances` to always keep it in order for better results.


.. py:function:: sample(rng_key: blackjax.types.PRNGKey, state: MFVIState, num_samples: int = 1)

   Sample from the mean-field approximation.


.. py:class:: meanfield_vi


   High-level implementation of Mean-Field Variational Inference.

   :param logdensity_fn: A function that represents the log-density function associated with
                         the distribution we want to sample from.
   :param optimizer: Optax optimizer to use to optimize the ELBO.
   :param num_samples: Number of samples to take at each step to optimize the ELBO.

   :rtype: A ``VIAlgorithm``.

   .. py:attribute:: init

      

   .. py:attribute:: step

      

   .. py:attribute:: sample

      


.. py:function:: generate_meanfield_logdensity(mu, rho)


