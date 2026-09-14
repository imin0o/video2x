# Shared processing settings, preserving M1 semantics.
import copy
import math

from art_processing_rng import ALGORITHM

MODEL_HASHES = ('1393f7c0e885f9d15a0668329a13f695ebd0ea45791f46d28145d7934824d224',
                '548a36f9c3f4ab8da56cd3b13badf23968bee207b396dad14d04b830e5f2ab2d')
DEFAULTS = dict(version=1, seed=0, weight_mode='noise', weight_strength=0.0,
                weight_layers=['Conv_16'], feature_mode='decay', feature_strength=0.0,
                feature_layer='PRelu_17', input_noise=0.0, time_mode='fixed', period=4.0,
                phase=0.0, modulation_depth=1.0, modulation_target='input_noise',
                input_blur=0.0, output_blur=0.0, passes=1, retain=0.0,
                source_color=0.0, luma_change=0.5)


def recipe(values):
    if not isinstance(values, dict) or values.keys() - DEFAULTS.keys():
        raise ValueError('Recipe must be an object with known M1 keys')
    result = copy.deepcopy(DEFAULTS | values)
    for key, low, high in [('weight_strength', 0, 2), ('feature_strength', 0, 1),
                           ('input_noise', 0, 128), ('period', 0.01, 3600),
                           ('phase', -3600, 3600), ('modulation_depth', 0, 1), ('input_blur', 0, 30),
                           ('output_blur', 0, 30), ('retain', 0, 1),
                           ('source_color', 0, 1), ('luma_change', 0, 1)]:
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be finite in [{low}, {high}]')
    for key, low, high in [('version', 1, 1), ('seed', 0, 2**64 - 1), ('passes', 1, 2)]:
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f'Invalid {key}')
    for key, choices in [('weight_mode', ('noise', 'decay')),
                         ('feature_mode', ('noise', 'decay', 'mask')),
                         ('time_mode', ('fixed', 'smooth', 'smooth-constant')),
                         ('modulation_target', ('input_noise',))]:
        if result[key] not in choices:
            raise ValueError(f'Unsupported {key}')
    if result['weight_mode'] == 'decay' and result['weight_strength'] > 1:
        raise ValueError('Weight decay must be <= 1')
    layers = result['weight_layers']
    if (not isinstance(layers, list) or not layers or any(not isinstance(x, str) for x in layers)
            or len(set(layers)) != len(layers)
            or set(layers) - {f'Conv_{i}' for i in range(0, 35, 2)}):
        raise ValueError('Select unique supported convolution layers')
    if result['feature_layer'] not in [f'PRelu_{i}' for i in range(1, 34, 2)]:
        raise ValueError('Feature target must be a supported 64-channel PReLU')
    return result


MODEL_ID = 'realesr-animevideov3-x2'


def configuration(values):
    """Migrate a flat M1 recipe or validate the portable, versioned M2 envelope."""
    if not isinstance(values, dict):
        raise ValueError('Recipe must be an object')
    if 'schema_version' not in values:
        settings = recipe(values)
    else:
        if set(values) != {'schema_version', 'method', 'model', 'rng', 'settings'}:
            raise ValueError('Unknown or missing M2 recipe keys')
        if type(values['schema_version']) is not int or values['schema_version'] != 2:
            raise ValueError('Unsupported recipe schema version')
        if values['method'] != 'm1-adopted-v1' or values['rng'] != ALGORITHM:
            raise ValueError('Unsupported processing or RNG semantics')
        expected = dict(id=MODEL_ID, param_sha256=MODEL_HASHES[0], binary_sha256=MODEL_HASHES[1])
        if values['model'] != expected:
            raise ValueError('Unsupported model identity or hashes')
        settings = recipe(values['settings'])
    return dict(schema_version=2, method='m1-adopted-v1', rng=ALGORITHM,
                model=dict(id=MODEL_ID, param_sha256=MODEL_HASHES[0], binary_sha256=MODEL_HASHES[1]),
                settings=settings)

