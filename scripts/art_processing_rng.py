"""Stateless named RNG streams, preserving every M1 seed and draw sequence."""
import hashlib

import numpy as np

ALGORITHM = 'm1-pcg64-v1'
DOMAINS = ('input', 'weight', 'feature', 'modulation', 'pass')


def derived_seed(seed, purpose):
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('Seed must be an unsigned 64-bit integer')
    if not isinstance(purpose, str) or purpose.split(':', 1)[0] not in DOMAINS:
        raise ValueError('Unknown RNG domain')
    digest = hashlib.sha256(f'{ALGORITHM}:{seed}:{purpose}'.encode()).digest()
    return int.from_bytes(digest[:16], 'little')


def rng(seed, purpose):
    # A fresh stream per logical target: consumption never leaks to another target/run.
    return np.random.Generator(np.random.PCG64(derived_seed(seed, purpose)))


def manifest(settings):
    purposes = ['input:0', 'input:1', 'modulation:0', 'pass:0', 'pass:1']
    purposes += ['weight:' + layer for layer in settings['weight_layers']]
    purposes += ['feature:' + settings['feature_layer']]
    return dict(algorithm=ALGORITHM, generator='NumPy PCG64',
                streams={p: str(derived_seed(settings['seed'], p)) for p in purposes},
                reserved=['modulation:0', 'pass:0', 'pass:1'])
