"""M1 model copies: typed weight mutation and Vulkan Scale feature intervention."""
from pathlib import Path

import numpy as np

from art_model_inspect import MODEL, inspect_model, sha256

from art_processing_config import MODEL_HASHES, recipe  # noqa: F401
from art_processing_rng import rng


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
        weight_strength = settings['weight_strength']
        if weight_strength == 0:
            continue
        dtype = '<f2' if tensor['dtype'] == 'fp16' else '<f4'
        offset, size = tensor['offset'], tensor['bytes']
        values = np.frombuffer(data[offset:offset + size], dtype=dtype).astype(np.float64)
        if settings['weight_mode'] == 'noise':
            values += (rng(settings['seed'], 'weight:' + tensor['layer']).standard_normal(values.size)
                       * weight_strength * tensor['rms'])  # An all-zero tensor remains zero.
        else:
            values *= 1 - weight_strength
        if not np.isfinite(values).all() or np.max(np.abs(values)) > np.finfo(dtype).max:
            raise ValueError('Mutated weight exceeds its finite encoding range')
        data[offset:offset + size] = values.astype(dtype).tobytes()
    text = original_param.read_text(encoding='utf-8')
    feature_strength = settings['feature_strength']
    if feature_strength:
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
            biases = random.standard_normal(64) * feature_strength
        elif settings['feature_mode'] == 'decay':
            scales *= 1 - feature_strength
        else:
            scales = (random.random(64) >= feature_strength).astype(float)
        prior = {x['name'] for x in inventory['layers'][:line_index - 1]}
        end = max(t['offset'] + t['bytes'] for t in inventory['tensors'] if t['layer'] in prior)
        data[end:end] = scales.astype('<f4').tobytes() + biases.astype('<f4').tobytes()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    base = directory / 'realesr-animevideov3'
    param, binary = directory / 'realesr-animevideov3-x2.param', directory / 'realesr-animevideov3-x2.bin'
    param.write_bytes(text.encode('utf-8') if feature_strength else original_param.read_bytes())
    binary.write_bytes(data)
    inspect_model(param, binary)
    return base, dict(param_sha256=sha256(param), binary_sha256=sha256(binary),
                      original_param_sha256=inventory['param_sha256'],
                      original_binary_sha256=inventory['binary_sha256'])
