"""Read-only inventory of the M0 RealESRGAN model and its serialized tensors."""
import argparse
import hashlib
import json
import math
import struct
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/realesrgan/realesr-animevideov3-x2'


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_model(param, binary):
    lines = param.read_text(encoding='utf-8').splitlines()
    if lines[0] != '7767517':
        raise ValueError('Unsupported ncnn param magic')
    layer_count, blob_count = map(int, lines[1].split())
    data = binary.read_bytes()
    offset, layers, tensors = 0, [], []

    def tensor(layer, role, count, tagged=False):
        nonlocal offset
        tag = struct.unpack_from('<I', data, offset)[0] if tagged else None
        if tagged:
            offset += 4
        if tag not in (None, 0, 0x01306B47):
            raise ValueError(f'Unsupported tensor encoding at {offset - 4}: {tag:#x}')
        dtype, width, code = ('fp16', 2, 'e') if tag == 0x01306B47 else ('fp32', 4, 'f')
        length = count * width
        chunk = data[offset:offset + length]
        if len(chunk) != length or count <= 0:
            raise ValueError('Truncated or empty tensor')
        values = [v[0] for v in struct.iter_unpack('<' + code, chunk)]
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f'Non-finite tensor in {layer}')
        tensors.append(dict(layer=layer, role=role, offset=offset, count=count,
                            dtype=dtype, bytes=length, sha256=hashlib.sha256(chunk).hexdigest(),
                            minimum=min(values), maximum=max(values),
                            rms=math.sqrt(sum(v * v for v in values) / count)))
        offset += (length + 3) // 4 * 4 if tagged else length

    for line in lines[2:]:
        if not line.strip():
            continue
        tokens = line.split()
        kind, name = tokens[:2]
        inputs, outputs = map(int, tokens[2:4])
        end = 4 + inputs + outputs
        params = dict(token.split('=', 1) for token in tokens[end:])
        entry = dict(type=kind, name=name, inputs=tokens[4:4 + inputs],
                     outputs=tokens[4 + inputs:end], params=params)
        if kind == 'Convolution':
            if int(params.get('8', 0)) or int(params.get('19', 0)):
                raise ValueError('INT8 or dynamic convolution not supported by this inspector')
            count, channels = int(params['6']), int(params['0'])
            kw, kh = int(params['1']), int(params.get('11', params['1']))
            if count % (channels * kw * kh):
                raise ValueError('Convolution shape does not match weight count')
            entry['weight_shape'] = [channels, count // (channels * kw * kh), kh, kw]
            tensor(name, 'weight', count, True)
            if int(params.get('5', 0)):
                tensor(name, 'bias', channels)
        elif kind == 'PReLU':
            tensor(name, 'slope', int(params['0']))
        elif kind not in ('Input', 'Split', 'PixelShuffle', 'Interp', 'BinaryOp'):
            raise ValueError(f'Unsupported layer: {kind}')
        layers.append(entry)
    if len(layers) != layer_count or offset != len(data):
        raise ValueError('Layer count or complete binary consumption check failed')
    return dict(param=str(param), binary=str(binary), param_sha256=sha256(param),
                binary_sha256=sha256(binary), layer_count=layer_count, blob_count=blob_count,
                bytes_consumed=offset, layer_types=dict(Counter(x['type'] for x in layers)),
                tensor_encodings=dict(Counter(x['dtype'] for x in tensors)),
                layers=layers, tensors=tensors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inspect_model(MODEL.with_suffix('.param'), MODEL.with_suffix('.bin'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(f"{report['layer_count']} layers; {len(report['tensors'])} tensors; {report['bytes_consumed']} bytes")


if __name__ == '__main__':
    main()
