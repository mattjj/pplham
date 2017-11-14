from itertools import repeat
from functools import partial
from contextlib import contextmanager
from collections import defaultdict
from tqdm import tqdm
import numpy as onp
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.tracer import trace, Node, toposort
from autograd.util import subvals
from autograd.wrap_util import unary_to_nary
from autograd.extend import primitive, Box, register_notrace, VJPNode, VSpace
from autograd.misc.flatten import flatten
from autograd.builtins import tuple

class ProbProgNode(Node):
  __slots__ = ['parents', 'logp', 'fun', 'sample']
  def __init__(self, ans, fun, args, kwargs, argnums, parents):
    args = subvals(args, zip(argnums, repeat(None)))
    full = lambda args_: subvals(args, zip(argnums, args_))
    if fun in logpdfs:
      self.logp = lambda x, args: logpdfs[fun](x, *full(args), **kwargs)
      self.sample = ans
    else:
      self.fun = lambda args: fun(*full(args), **kwargs)
      self.sample = None
    self.parents = parents

  def initialize_root(self):
    self.parents = []
    self.fun = lambda args: None
    self.sample = None

def make_logp(fun, *args, **kwargs):
  def fun_(rng):
    with use_rng(rng): return tuple(fun(*args, **kwargs))
  start_node = ProbProgNode.new_root()
  out, end_node = trace(start_node, fun_, global_rng)
  graph = list(toposort(end_node))[::-1]
  rvs = {n: n.sample for n in graph if n.sample is not None}
  def logpdf(rvs, obs, *args):
    vals = merge(rvs, zip(end_node.parents, obs))
    logp = 0.
    for node in graph:
      parent_vals = (vals[p] for p in node.parents)
      if node.sample is None:
        vals[node] = node.fun(parent_vals)
      else:
        logp += node.logp(vals[node], parent_vals)
    return logp
  return rvs, out, logpdf

def merge(dct, pairs):
  return dict(dct, **{k: v for k, v in pairs if v is not None})

### setting up a global rng and rng primitives

class RNG(object): pass
class RNGBox(Box): pass
RNGBox.register(RNG)
global_rng = RNG()

@contextmanager
def use_rng(rng):
  global global_rng
  global_rng, prev_rng = rng, global_rng
  yield
  global_rng = prev_rng

### core for rng primitives and their densities

def rng_primitive(fun):
  @primitive
  def _fun(rng, *args, **kwargs):
    return fun(*args, **kwargs)
  def wrapped(*args, **kwargs):
    return _fun(global_rng, *args, **kwargs)
  wrapped._fun = _fun
  return wrapped

no_density = lambda *args, **kwargs: 0.
logpdfs = defaultdict(lambda: no_density)

def deflogp(fun, logp):
  def _logp(val, rng, *args, **kwargs):
    return logp(val, *args, **kwargs)
  logpdfs[fun._fun] = _logp

### rng functions

normal = rng_primitive(onp.random.normal)

def randn_logp(x, loc=0., scale=1., size=None):
    return -1./2 * (np.sum((x - loc)**2 / scale**2)
                    + np.sum(np.log(scale**2)) + np.size(x) * np.log(2*np.pi))
deflogp(normal, randn_logp)


### basic examples

def sample_prior(A):
  z = normal(np.ones(2), 2 * np.ones(2))
  x = normal(np.dot(A, z), 3 * np.ones(2))
  return z, x

sample = lambda: sample_prior(np.array([[1., 0.], [0., 0.]]))
rvs, prior_sample, logp = make_logp(sample)
print rvs
print prior_sample
print logp(rvs, prior_sample)
print

obs = (None, np.ones(2))
print logp(rvs, obs)
print grad(logp)(rvs, obs)

# ### hmc

# def _hmc_transition(logp, x0, step_size, num_steps):
#   def leapfrog_step(x, v):
#     v = v + step_size/2. * grad(logp)(x)
#     x = x + step_size * v
#     v = v + step_size/2. * grad(logp)(x)
#     return x, v

#   def accept(x0, v0, x1, v1):
#     thresh = min(0., energy(x1, v1) - energy(x0, v0))
#     return np.log(npr.rand()) < thresh

#   def energy(x, v): return logp(x) - 0.5 * np.dot(v, v)

#   v0 = npr.normal(size=x0.shape)
#   x, v = x0, v0
#   for i in xrange(num_steps):
#     x, v = leapfrog_step(x, v)
#   return x if accept(x0, v0, x, v) else x0

# def hmc_transition(logp, x0, *args):
#   _x0, unflatten = flatten(x0)
#   _logp = lambda x: logp(unflatten(x))
#   return unflatten(_hmc_transition(_logp, _x0, *args))

# def posterior_inference(sampler, obs, step_size, num_steps, num_iters):
#   latents, _,  logp = make_logp(sampler)
#   _logp = lambda latents: logp(latents, obs)
#   transition = lambda z: hmc_transition(_logp, z, step_size, num_steps)

#   samples = [latents]
#   for _ in tqdm(range(num_iters)):
#     samples.append(transition(samples[-1]))
#   return samples

# def test0():
#   """Sampling from a Gaussian prior."""

#   # define model by writing a sampler function
#   def sampler_fun():
#     z = normal(0., np.array([1., 4.]))
#     return normal(z)

#   # set an observation
#   obs = np.ones(2)

#   # run posterior inference
#   # TODO samples comes out as a list of dicts, each keyed by ProbProgNodes.
#   # That's a mess! Gotta improve that. Name the variables that we want to do
#   # inference over?
#   samples = posterior_inference(sampler_fun, obs, 0.1, 25, 100)
#   print samples

#   # # plot results
#   # fig, ax = plt.subplots()
#   # ax.plot(samples[:,0], samples[:,1], 'k.')
#   # ax.axis('equal')
#   # ax.set_title(r'Samples from $x \sim \mathcal{N}([0, 0], [1, 4])$')

# if __name__ == '__main__':
#   test0()
