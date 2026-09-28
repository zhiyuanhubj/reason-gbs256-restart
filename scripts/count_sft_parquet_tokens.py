#!/usr/bin/env python
"""Count complete Qwen3.5/SWIFT SFT input tokens without materializing image tensors.

Text is encoded with the real training template. Every image header is read and
its visual grid uses the same qwen_vl_utils smart_resize as the training loader.
Validate the compact counting path against full processor encoding before a scan.
"""

import argparse
from collections import Counter
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time
from types import MethodType

SPECIAL_TOKENS = [
    '<|action_start|>', '<|action_end|>', '<|action_sep|>',
    '<|thought_start|>', '<|thought_end|>',
]
ROLE_MAP = {'system': 'system', 'human': 'user', 'user': 'user',
            'gpt': 'assistant', 'assistant': 'assistant'}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def image_bytes(image):
    if isinstance(image, dict):
        assert image.get('bytes'), 'Expected embedded image bytes'
        return image['bytes']
    assert isinstance(image, bytes), f'Unexpected image representation: {type(image)}'
    return image


def messages(row):
    result = []
    for item in row['conversations']:
        role = item.get('from') or item.get('role')
        assert role in ROLE_MAP, f'Unsupported conversation role: {role}'
        content = item.get('value') or item.get('content') or ''
        assert isinstance(content, str)
        result.append({'role': ROLE_MAP[role], 'content': content})
    assert result and any(x['role'] == 'assistant' for x in result)
    return result


def make_template(processor):
    from swift import get_template
    template = get_template(processor, max_length=10**9, truncation_strategy='raise',
                            padding_free=True, sequence_parallel_size=4,
                            add_non_thinking_prefix=False)
    template.set_mode('train')
    assert type(template).__name__ == 'Qwen3_5Template'
    assert template.max_pixels is None, 'Unexpected additional SWIFT image rescaling'
    return template


def compact_template(processor):
    from swift.template.base import Template
    template = make_template(processor)
    # Keep SWIFT conversation preparation/tokenization; omit pixel materialization
    # and repeated image_pad expansion, replacing those lengths analytically.
    template._load_image = lambda image, load_images: image

    def replace_tag(self, media_type, index, inputs):
        assert media_type == 'image', 'This counter only supports image SFT'
        return ['<|vision_start|><|image_pad|><|vision_end|>']

    template.replace_tag = MethodType(replace_tag, template)
    template._encode = MethodType(Template._encode, template)
    return template


def count_row(row, template, processor, grid_cache):
    from PIL import Image
    from qwen_vl_utils.vision_process import smart_resize, SPATIAL_MERGE_SIZE
    patch = processor.image_processor.patch_size
    merge = processor.image_processor.merge_size
    assert merge == SPATIAL_MERGE_SIZE
    dims = Counter()
    visual = 0
    for image in row['images']:
        with Image.open(BytesIO(image_bytes(image))) as pil:
            width, height = pil.size
        dims[f'{width}x{height}'] += 1
        key = (width, height)
        if key not in grid_cache:
            h, w = smart_resize(height, width, factor=patch * merge)
            assert h % (patch * merge) == 0 and w % (patch * merge) == 0
            grid_cache[key] = (h // patch) * (w // patch) // (merge**2)
        visual += grid_cache[key]
    encoded = template.encode({'messages': messages(row), 'images': [None] * len(row['images'])})
    ids = encoded['input_ids']
    placeholders = ids.count(template.image_token_id)
    assert placeholders == len(row['images']), 'Image placeholders do not match embedded frames'
    text = len(ids) - placeholders
    return {'text_tokens': text, 'visual_tokens': visual, 'tokens': text + visual,
            'images': len(row['images']), 'dimensions': dims, 'compact_ids': ids}


def validate(args, manifest, processor, compact):
    import pyarrow.parquet as pq
    from PIL import Image
    from qwen_vl_utils import fetch_image
    full = make_template(processor)
    cases = []
    grid_cache = {}
    # Two whole samples per dataset, chosen from distinct end shards.
    for name, spec in manifest.items():
        files = sorted(Path(spec['path']).glob('*.parquet'))
        for path in [files[0], files[-1]]:
            row = next(pq.ParquetFile(path).iter_batches(batch_size=1)).to_pylist()[0]
            fast = count_row(row, compact, processor, grid_cache)
            actual = full.encode({'messages': messages(row),
                                  'images': [image_bytes(x) for x in row['images']]})
            ids = actual['input_ids']
            visual = ids.count(full.image_token_id)
            assert len(ids) == fast['tokens'] and visual == fast['visual_tokens']
            assert [i for i in ids if i != full.image_token_id] == [
                i for i in fast['compact_ids'] if i != full.image_token_id]
            cases.append({'dataset': name, 'file': path.name, 'images': fast['images'],
                          'tokens': len(ids), 'visual_tokens': visual,
                          'exact_nonvisual_token_ids_match': True})
            print(f'[validated] {name} {path.name}: {len(ids)} tokens', flush=True)
            del actual, ids
    # Exercise the resize boundaries too, beyond resolutions present in samples.
    resize_cases = []
    from qwen_vl_utils.vision_process import smart_resize
    patch = processor.image_processor.patch_size
    merge = processor.image_processor.merge_size
    for width, height in [(32, 32), (123, 257), (1280, 720), (4096, 4096), (6000, 4000)]:
        image = Image.new('RGB', (width, height))
        resized = fetch_image({'image': image}, image_patch_size=patch)
        output = processor.image_processor(images=[resized], return_tensors='pt', do_resize=False)
        actual_grid = output['image_grid_thw'][0].tolist()
        h, w = smart_resize(height, width, factor=patch * merge)
        assert actual_grid == [1, h // patch, w // patch]
        resize_cases.append({'width': width, 'height': height, 'grid': actual_grid})
        del image, resized, output
    save_json(Path(args.output_dir) / 'validation.json', {
        'model': args.model, 'cases': cases, 'resize_cases': resize_cases, 'status': 'passed'})


def scan(args, manifest, processor, compact, rank, world):
    import pyarrow.parquet as pq
    output = Path(args.output_dir)
    jobs = []
    for name, spec in manifest.items():
        files = sorted(Path(spec['path']).glob('*.parquet'))
        assert len(files) == spec['shards']
        jobs.extend((name, path) for path in files)
    # Balance shards by file size rather than putting every large Genshin shard
    # on the same rank. Each original Parquet is read by exactly one process.
    assigned = [[] for _ in range(world)]
    loads = [0] * world
    for job in sorted(jobs, key=lambda x: x[1].stat().st_size, reverse=True):
        owner = min(range(world), key=lambda r: loads[r])
        assigned[owner].append(job)
        loads[owner] += job[1].stat().st_size
    grid_cache = {}
    started = time.time()
    for name, path in assigned[rank]:
        destination = output / 'shards' / name / (path.name + '.json')
        stats = {'dataset': name, 'source': str(path), 'source_bytes': path.stat().st_size,
                 'rows': 0, 'images': 0, 'text_tokens': 0, 'visual_tokens': 0, 'tokens': 0,
                 'overlength_rows': 0, 'overlength_tokens': 0, 'max_length': 0,
                 'dimensions': Counter()}
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1, columns=['images', 'conversations'], use_threads=False):
            row = batch.to_pylist()[0]
            counts = count_row(row, compact, processor, grid_cache)
            stats['rows'] += 1
            for key in ('images', 'text_tokens', 'visual_tokens', 'tokens'):
                stats[key] += counts[key]
            stats['dimensions'].update(counts['dimensions'])
            stats['max_length'] = max(stats['max_length'], counts['tokens'])
            if counts['tokens'] > args.max_train_length:
                stats['overlength_rows'] += 1
                stats['overlength_tokens'] += counts['tokens']
            if stats['rows'] % 100 == 0:
                save_json(output / 'progress' / f'rank-{rank:02d}.json', {
                    'dataset': name, 'shard': path.name, 'rows': stats['rows'],
                    'elapsed_seconds': time.time() - started})
        assert stats['rows'] == parquet.metadata.num_rows
        assert stats['tokens'] == stats['visual_tokens'] + stats['text_tokens']
        save_json(destination, stats)
        print(f'[shard] rank={rank} {name}/{path.name}: {stats["rows"]} rows, {stats["tokens"]} tokens', flush=True)
    save_json(output / f'rank-{rank:02d}.done.json', {'rank': rank, 'world': world, 'shards': len(assigned[rank])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--max-train-length', type=int, default=131072)
    args = parser.parse_args()
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '16384')
    import torch
    from swift import get_processor
    torch.set_num_threads(1)
    manifest = json.loads(Path(args.manifest).read_text())
    processor = get_processor(args.model, new_special_tokens=SPECIAL_TOKENS)
    compact = compact_template(processor)
    rank = int(os.environ.get('SLURM_PROCID', '0'))
    world = int(os.environ.get('SLURM_NTASKS', '1'))
    if args.validate_only:
        assert world == 1
        validate(args, manifest, processor, compact)
    else:
        validation = json.loads((Path(args.output_dir) / 'validation.json').read_text())
        assert validation['status'] == 'passed' and validation['model'] == args.model
        if rank == 0:
            import transformers
            from qwen_vl_utils import vision_process
            tokenizer_path = Path(args.model) / 'tokenizer.json'
            save_json(Path(args.output_dir) / 'method.json', {
                'model': args.model, 'tokenizer_sha256': hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
                'transformers_version': transformers.__version__, 'template': type(compact).__name__,
                'patch_size': processor.image_processor.patch_size,
                'merge_size': processor.image_processor.merge_size,
                'image_min_token_num': vision_process.IMAGE_MIN_TOKEN_NUM,
                'image_max_token_num': vision_process.IMAGE_MAX_TOKEN_NUM,
                'special_tokens': SPECIAL_TOKENS, 'add_non_thinking_prefix': False,
                'text_includes_chat_and_vision_boundary_tokens': True,
                'all_samples_counted_once_without_truncation_or_deduplication': True,
                'visual_count_method': 'Every embedded image header; exact training smart_resize and merged grid',
                'text_count_method': 'Real SWIFT training template input IDs excluding image_pad placeholders',
                'world': world})
        scan(args, manifest, processor, compact, rank, world)


if __name__ == '__main__':
    main()
