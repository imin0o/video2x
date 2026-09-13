"""M1 model copies: typed weight mutation and Vulkan Scale feature intervention."""
import hashlib
import math
from pathlib import Path

import numpy as np

from art_model_inspect import MODEL, inspect_model, sha256

MODEL_HASHES = ('1393f7c0e885f9d15a0668329a13f695ebd0ea45791f46d28145d7934824d224',
                '548a36f9c3f4ab8da56cd3b13badf23968bee207b396dad14d04b830e5f2ab2d')
DEFAULTS = dict(version=1, seed=0, weight_mode='noise', weight_strength=0.0,
                weight_layers=['Conv_16'], feature_mode='decay', feature_strength=0.0,
                feature_layer='PRelu_17', input_noise=0.0, time_mode='fixed', period=4.0,
                phase=0.0, input_blur=0.0, output_blur=0.0, passes=1, retain=0.0)


def rng(seed, purpose):
    digest = hashlib.sha256(f'm1-pcg64-v1:{seed}:{purpose}'.encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], 'little')))


def recipe(values):
    if not isinstance(values, dict) or values.keys() - DEFAULTS.keys():
        raise ValueError('Recipe must be an object with known M1 keys')
    result = DEFAULTS | values
    for key, low, high in [('weight_strength', 0, 2), ('feature_strength', 0, 1),
                           ('input_noise', 0, 128), ('period', 0.01, 3600),
                           ('phase', -3600, 3600), ('input_blur', 0, 30),
                           ('output_blur', 0, 30), ('retain', 0, 1)]:
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be finite in [{low}, {high}]')
    for key, low, high in [('version', 1, 1), ('seed', 0, 2**64 - 1), ('passes', 1, 2)]:
        if type(result[key]) is not int or not low <= result[key] <= high:
            raise ValueError(f'Invalid {key}')
    for key, choices in [('weight_mode', ('noise', 'decay')),
                         ('feature_mode', ('noise', 'decay', 'mask')),
                         ('time_mode', ('fixed', 'smooth'))]:
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


def model_copy(directory, settings):
    """Always derive from immutable original bytes; never mutate an earlier copy."""
    original_param, original_bin = MODEL.with_suffix('.param'), MODEL.with_suffix('.bin')
    inventory = inspect_model(original_param, original_bin)
    if (inventory['param_sha256'], inventory['binary_sha256']) != MODEL_HASHES:
        raise ValueError('M1 supports only the recorded animevideov3-x2 model hashes')
    data = bytearray(original_bin.read_bytes())
    for tensor in inventory['tensors']:
        if tensor['role'] != 'weight' or tensor['layer'] not in settings['weight_layers']:
            continue
        strength = settings['weight_strength']
        if strength == 0:
            continue
        dtype = '<f2' if tensor['dtype'] == 'fp16' else '<f4'
        offset, size = tensor['offset'], tensor['bytes']
        values = np.frombuffer(data[offset:offset + size], dtype=dtype).astype(np.float64)
        if settings['weight_mode'] == 'noise':
            values += (rng(settings['seed'], 'weight:' + tensor['layer']).standard_normal(values.size)
                       * strength * tensor['rms'])  # An all-zero tensor remains zero.
        else:
            values *= 1 - strength
        if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(dtype).max:
            raise ValueError('Mutated weight exceeds its finite encoding range')
        data[offset:offset + size] = values.astype(dtype).tobytes()
    text = original_param.read_text(encoding='utf-8')
    strength = settings['feature_strength']
    if strength:
        # A real intermediate operation after PReLU, before the following convolution.
        # Channel noise/masks are spatially constant, including tile overlap/padding.
        target = next(x for x in inventory['layers'] if x['name'] == settings['feature_layer'])
        blob = target['outputs'][0]
        lines = text.splitlines()
        layer_count, blob_count = map(int, lines[1].split())
        lines[1] = f'{layer_count + 1} {blob_count + 1}'
        line_index = next(i for i, line in enumerate(lines[2:], 2)
                          if line.split()[1] == target['name'])
        tokens = lines[line_index].split()
        tokens[5] = 'm1_feature_input'
        lines[line_index] = ' '.join(tokens)
        lines.insert(line_index + 1, f'Scale m1_feature 1 1 m1_feature_input {blob} 0=64 1=1')
        text = '\n'.join(lines) + '\n'
        scales, biases = np.ones(64), np.zeros(64)
        random = rng(settings['seed'], 'feature:' + target['name'])
        if settings['feature_mode'] == 'noise':
            biases = random.standard_normal(64) * strength
        elif settings['feature_mode'] == 'decay':
            scales *= 1 - strength
        else:
            scales = (random.random(64) >= strength).astype(float)
        prior = {x['name'] for x in inventory['layers'][:line_index - 1]}
        end = max(t['offset'] + t['bytes'] for t in inventory['tensors'] if t['layer'] in prior)
        data[end:end] = scales.astype('<f4').tobytes() + biases.astype('<f4').tobytes()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    base = directory / 'realesr-animevideov3'
    param, binary = directory / 'realesr-animevideov3-x2.param', directory / 'realesr-animevideov3-x2.bin'
    param.write_bytes(text.encode('utf-8') if strength else original_param.read_bytes())
    binary.write_bytes(data)
    inspect_model(param, binary)
    return base, dict(param_sha256=sha256(param), binary_sha256=sha256(binary),
                      original_param_sha256=inventory['param_sha256'],
                      original_binary_sha256=inventory['binary_sha256'])
