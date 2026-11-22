"""Write a synthetic rank-1 LoRA adapter (GGUF) for a base model, to test llama-server's LoRA path.
Usage: make_lora.py <base.gguf> <out.gguf> <std>
Targets every blk.N.ffn_down.weight. lora_a and lora_b are random normal with the given std (seed 0);
alpha = rank = 1, so the weight change is scale * B @ A."""
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter

base, out, std = sys.argv[1], sys.argv[2], float(sys.argv[3])
r = GGUFReader(base)
arch = bytes(r.fields['general.architecture'].parts[r.fields['general.architecture'].data[0]]).decode()
rng = np.random.default_rng(0)
w = GGUFWriter(out, arch)
w.add_string('general.type', 'adapter')
w.add_string('adapter.type', 'lora')
w.add_float32('adapter.lora.alpha', 1.0)
n = 0
for t in r.tensors:
    if not t.name.endswith('ffn_down.weight'):
        continue
    n_in, n_out = int(t.shape[0]), int(t.shape[1])  # ggml order: ne[0] = input width
    a = (rng.standard_normal((1, n_in)) * std).astype(np.float32)   # ggml ne = [n_in, 1]
    b = (rng.standard_normal((n_out, 1)) * std).astype(np.float32)  # ggml ne = [1, n_out]
    w.add_tensor(t.name + '.lora_a', a)
    w.add_tensor(t.name + '.lora_b', b)
    n += 1
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f'{out}: arch {arch}, {n} ffn_down tensors, rank 1, std {std}')
