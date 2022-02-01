import functools
from typing import Callable, Union

import jax

from blackjax import hmc, nuts
from blackjax.base import AdaptationAlgorithm
from blackjax.stan_warmup import window_adaptation_base, window_adaptation_schedule
from blackjax.types import Array, PRNGKey, PyTree


def window_adaptation(
    algorithm: Union[hmc, nuts],
    logprob_fn: Callable,
    is_mass_matrix_diagonal: bool = True,
    initial_step_size: float = 1.0,
    target_acceptance_rate: float = 0.65,
    **parameters,
) -> AdaptationAlgorithm:

    kernel_fn = algorithm.new_kernel()

    def kernel(rng_key, state, step_size, inverse_mass_matrix):
        return kernel_fn(
            rng_key, state, logprob_fn, step_size, inverse_mass_matrix, **parameters
        )

    def init_fn(position: PyTree):
        return algorithm.init(position, logprob_fn)

    def run(rng_key: PRNGKey, position: PyTree, num_steps: int = 1000):

        init_state = init_fn(position)
        schedule_fn = window_adaptation_schedule(num_steps)
        init, update, final = window_adaptation_base(
            kernel,
            schedule_fn,
            is_mass_matrix_diagonal,
            target_acceptance_rate=target_acceptance_rate,
        )

        def one_step(carry, rng_key):
            state, warmup_state = carry
            state, warmup_state, _ = update(rng_key, state, warmup_state)
            return (state, warmup_state)
        

        warmup_state = init(rng_key, init_state, initial_step_size)
        step_size, inverse_mass_matrix = final(warmup_state)
        kernel_last = functools.partial(
             kernel, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix
        )

        return (kernel_last, init_state, warmup_state, one_step)
        # last_state, warmup_chain = jax.lax.scan(
        #     one_step,
        #     (init_state, warmup_state),
        #     keys,
        # )
        # last_chain_state, last_warmup_state = last_state

        # step_size, inverse_mass_matrix = final(last_warmup_state)
        # kernel_last = functools.partial(
        #     kernel, step_size=step_size, inverse_mass_matrix=inverse_mass_matrix
        # )

        # return last_chain_state, kernel_last, (step_size, inverse_mass_matrix)

    return AdaptationAlgorithm(run)
