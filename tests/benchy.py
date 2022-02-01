import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.util import initialize_model
from numpyro.infer import MCMC, NUTS
import pandas as pd

import blackjax
import blackjax.diagnostics as diagnostics


def inference_loop(kernel, num_samples, rng_key, initial_state):
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


# Data of the Eight Schools Model
J = 8
y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

# Eight Schools example - Non-centered Reparametrization
def eight_schools_noncentered(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        with numpyro.handlers.reparam(config={"theta": TransformReparam()}):
            theta = numpyro.sample(
                "theta",
                dist.TransformedDistribution(
                    dist.Normal(0.0, 1.0), dist.transforms.AffineTransform(mu, tau)
                ),
            )
        numpyro.sample("obs", dist.Normal(theta, sigma), obs=y)

rng_key = jax.random.PRNGKey(0)

init_params, potential_fn_gen, *_ = initialize_model(
    rng_key,
    eight_schools_noncentered,
    model_args=(J, sigma, y),
    dynamic_args=True,
)

initial_position = init_params.z
logprob = jax.jit(lambda position: -potential_fn_gen(J, sigma, y)(position))

num_warmup_steps = 1_000
num_sampling_steps = 100_000


warmup_key, inference_key = jax.random.split(rng_key, 2)

warmup = blackjax.window_adaptation(
    algorithm=blackjax.nuts,
    logprob_fn=logprob,
    target_acceptance_rate=0.8,
)


tic1 = pd.Timestamp.now()
kernel, init_st, warmup_st, step = warmup.run(
    warmup_key, initial_position, num_warmup_steps
)
tic2 = pd.Timestamp.now()
print("Runtime for Blackjax's stan warmup init", tic2 - tic1)

def one_step(carry, rng_key):
    state, warmup_state = carry
    state, _ = kernel(rng_key, state)
    return state, warmup_state

@jax.jit
def body_i(i, state):
    return jax.lax.cond(
        i < num_warmup_steps,
        lambda x: step((x[0], x[1]), rng_key),
        lambda x: one_step((x[0], x[1]), rng_key),
        (init_st, warmup_st, rng_key),
    )

tic1 = pd.Timestamp.now()
jax.lax.fori_loop(
    0,
    num_warmup_steps + num_sampling_steps,
    body_i,
    (init_st, warmup_st),
)
tic2 = pd.Timestamp.now()
print("Runtime for Blackjax's sampling (warmup + sampling)", tic2 - tic1)

tic1 = pd.Timestamp.now()
nuts_kernel = NUTS(
    eight_schools_noncentered,
    target_accept_prob=0.8,
)
mcmc = MCMC(
    nuts_kernel,
    num_warmup=num_warmup_steps,
    num_samples=num_sampling_steps,
    progress_bar=False,
)
mcmc.run(rng_key, J, sigma, y=y)
samples = mcmc.get_samples()
tic2 = pd.Timestamp.now()
print("Runtime for numpyro's NUTS warmup + sampling", tic2 - tic1)
