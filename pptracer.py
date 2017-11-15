from tqdm import tqdm
import numpy as onp
from numpy.random import RandomState

import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import expit, logit
from autograd import grad

from autograd.tracer import trace, Node, toposort
from autograd.util import subvals, func
from autograd.extend import primitive, Box
from autograd.misc.flatten import flatten
from autograd.builtins import tuple

class ProbProgNode(Node):
  __slots__ = ['parents', 'is_rv', 'x', 'f']
  def __init__(self, ans, fun, args, kwargs, argnums, parents):
    def f(x, logp):
      _args = subvals(args, zip(argnums, (p.x for p in parents)))
      if self.is_rv:
        return x, logp + logpdfs[fun](x, *_args[1:], **kwargs)
      else:
        return fun(*_args, **kwargs), logp
    self.parents = parents
    self.is_rv = fun in logpdfs
    self.x = ans
    self.f = f

  def initialize_root(self):
    self.parents = []
    self.is_rv = False
    self.f = lambda x, logp: (x, logp)
    self.x = None

def make_logp(fun, ifix):
  def fun_(rng): return tuple(fun(rng))
  start_node = ProbProgNode.new_root()
  _, end_node = trace(start_node, fun_, RNG())
  graph = list(toposort(end_node))[::-1]
  xnodes = [end_node.parents[i] for i in ifix]
  znodes = [node for node in graph if node.is_rv and node not in xnodes]
  def logpdf(z, x):
    rvs = dict(zip(znodes, z) + zip(xnodes, x))
    logp = 0.
    for node in graph:
      node.x, logp = node.f(rvs.get(node), logp)
    return logp
  def zfilt(zs):
    rvs = dict(zip(znodes, zs))
    return [rvs[node] for node in end_node.parents if node in rvs]
  return logpdf, [node.x for node in znodes], zfilt


### rng primitives

class RNG(RandomState):
  def bernoulli(self, logits):
    return 0 + (logits > logit(self.uniform(size=len(logits))))

class RNGBox(Box):
  @primitive
  def normal(self, *args, **kwargs):
    return self.normal(*args, **kwargs)

  @primitive
  def bernoulli(self, *args, **kwargs):
    return self.bernoulli(*args, **kwargs)

RNGBox.register(RNG)

logpdfs = {}
def deflogp(fun, logp):
  logpdfs[fun] = logp

def normal_logp(x, loc=0., scale=1., size=None):
  return -1./2 * (np.sum((x - loc)**2 / scale**2)
                  + np.sum(np.log(scale**2)) + np.size(x) * np.log(2*np.pi))
deflogp(func(RNGBox.normal), normal_logp)

def bernoulli_logp(x, logits, size=None):
  return -np.sum(np.logaddexp(0., np.where(x, -1., 1.) * logits))
deflogp(func(RNGBox.bernoulli), bernoulli_logp)

### hmc

def _hmc_transition(logp, x0, step_size, num_steps):
  def leapfrog_step(x, v):
    v = v + step_size/2. * grad(logp)(x)
    x = x + step_size * v
    v = v + step_size/2. * grad(logp)(x)
    return x, v

  def accept(x0, v0, x1, v1):
    thresh = min(0., energy(x1, v1) - energy(x0, v0))
    return np.log(npr.rand()) < thresh

  def energy(x, v): return logp(x) - 0.5 * np.dot(v, v)

  v0 = npr.normal(size=x0.shape)
  x, v = x0, v0
  for i in xrange(num_steps):
    x, v = leapfrog_step(x, v)
  return x if accept(x0, v0, x, v) else x0

def hmc_transition(logp, x0, *args):
  _x0, unflatten = flatten(x0)
  _logp = lambda x: logp(unflatten(x))
  return unflatten(_hmc_transition(_logp, _x0, *args))

def posterior_inference(sampler, observed, step_size, num_steps, num_iters):
  if all(x is None for x in observed):
    ifix, x = [], []
  else:
    ifix, x = zip(*[(i, x) for i, x in enumerate(observed) if x is not None])
  logp, z, zfilt = make_logp(sampler, ifix)
  logp_ = lambda z: logp(z, x)
  transition = lambda z: hmc_transition(logp_, z, step_size, num_steps)

  samples = [zfilt(z)]
  for _ in tqdm(range(num_iters)):
    z = transition(z)
    samples.append(zfilt(z))
  return samples

def test0():
  npr.seed(0)

  # define model by writing a sampler function
  def sampler_fun():
    z = normal(0., np.array([1., 4.]))
    x = normal(z, np.array([1., 2.]))
    return z, x

  # set up observations
  observations = (None, np.ones(2))

  # run posterior inference
  samples = posterior_inference(sampler_fun, observations, 0.1, 25, 100)
  samples = np.vstack(samples)

  # plot results
  fig, ax = plt.subplots()
  ax.plot(samples[:,0], samples[:,1], 'k.')
  ax.axis('equal')
  ax.set_title(r'Samples from $x \sim \mathcal{N}([0, 0], [1, 4])$')

def test1():
  npr.seed(0)
  N = 100
  D = 5

  # generate some synth data
  x = npr.randn(N, D)
  beta = npr.randn(D)
  y = RNG().bernoulli(np.dot(x, beta))

  # write da model
  def sampler_fun(rng):
    beta = rng.normal(np.zeros(D), np.ones(D))
    y = rng.bernoulli(np.dot(x, beta))
    return beta, y

  # set up observations
  observations = (None, y)

  # run posterior inference
  samples = posterior_inference(sampler_fun, observations, 1. / N, 10, 250)
  samples = np.vstack(samples)

  print beta
  print samples[-10:].mean(0)

if __name__ == '__main__':

  ### basic example

  npr.seed(0)

  def sample_prior(rng):
    A = np.array([[1., 0.], [0., 0.]])
    z1 = rng.normal(np.ones(2), 2 * np.ones(2))
    z2 = rng.normal(np.zeros(2))
    x = rng.normal(np.dot(A, z1) + z2, 3 * np.ones(2))
    return z1, x

  logp, latent_rvs, zfilt = make_logp(sample_prior, (1,))
  print latent_rvs         # should be (z1, z2)
  print zfilt(latent_rvs)  # should be just z1
  print logp(latent_rvs, np.ones(2))
  print

  # test0()
  test1()
  plt.show()
