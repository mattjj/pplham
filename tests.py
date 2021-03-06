import autograd.numpy as np

from pptracer import make_logp, posterior_inference, RNG

N = 3
D = 2000

def assert_closeish(x, y):
  assert np.abs(x - y) < 0.1, '{} not closeish to {}'.format(x, y)

def gg_sampler(rng):
  z = rng.normal(np.zeros(D), np.ones(D))
  x = rng.normal(np.tile(z, (N, 1)), np.ones((N, D)))
  return x, z

def test_prior_samples():
  observations = (None, None)
  samples = posterior_inference(gg_sampler, observations,
                                0.1, 10, 50)
  x, z = samples[-1]

  assert x.shape == (N, D)
  assert_closeish(np.mean(x), 0.0)
  assert_closeish(np.var(x),  2.0)

  assert z.shape == (D,)
  assert_closeish(np.mean(z), 0.0)
  assert_closeish(np.var(z),  1.0)

def test_posterior_samples():
  rng = RNG()

  x = np.tile(rng.normal(np.ones((N, 1)), np.ones((N, 1))), (1, D))
  observations = (x, None)
  samples = posterior_inference(gg_sampler, observations,
                                0.1, 10, 50)
  z, = samples[-1]
  assert z.shape == (D,)

  z_correct_mean = np.sum(x[:, 0]) / (1 + N)
  z_correct_var  = 1.0 / (N + 1)

  assert_closeish(np.mean(z), z_correct_mean)
  assert_closeish(np.var(z) , z_correct_var)
