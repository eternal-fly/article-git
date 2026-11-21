"""CUDA execution of the frozen exact-suffix circuit, without protocol changes.

Gate matrices and diagonal phases are built by the original CPU/Torch functions.
CUDA changes only where state-vector arithmetic is executed. All 20 suffixes
remain explicit; final/TM3 expectations use double-precision Walsh reductions.
"""
from grid_common import *
import time
import torch
from hamiltonian_dtc_qrc import (DTCQRCConfig, generate_dtc_params,
    build_dtc_diagonal_phase, rx_gate, hamiltonian_modulation_gates)

# The frozen CPU common module masks CUDA on import. Undo that execution setting
# here, before CuPy initializes CUDA; never alter the frozen scientific sources.
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ.setdefault("CUPY_CACHE_DIR", str(HERE / ".cupy_cache"))
import cupy as cp

CUDA = r'''
extern "C" __global__ void gate(float2* state, const float2* gates,
    const float2* phase, int active, int n, int t, int q, int dim,
    int stride, int event, int wire, float c, float s, int diagonal) {
    long k = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= (long)active * (dim/2)) return;
    int batch = k/(dim/2), pair = k%(dim/2);
    int a = (pair/stride)*(2*stride)+pair%stride, b=a+stride;
    long offset = (long)batch*dim;
    float2 x=state[offset+a], y=state[offset+b], u, v;
    if (event >= 0) {
        const float2* z = gates+(((long)(batch%n)*t+event)*q+wire)*4;
        u.x=z[0].x*x.x-z[0].y*x.y+z[1].x*y.x-z[1].y*y.y;
        u.y=z[0].x*x.y+z[0].y*x.x+z[1].x*y.y+z[1].y*y.x;
        v.x=z[2].x*x.x-z[2].y*x.y+z[3].x*y.x-z[3].y*y.y;
        v.y=z[2].x*x.y+z[2].y*x.x+z[3].x*y.y+z[3].y*y.x;
    } else {
        u=make_float2(c*x.x+s*y.y, c*x.y-s*y.x);
        v=make_float2(c*y.x+s*x.y, c*y.y-s*x.x);
    }
    if (diagonal) {
        float2 p=phase[a], r=phase[b];
        u=make_float2(u.x*p.x-u.y*p.y, u.x*p.y+u.y*p.x);
        v=make_float2(v.x*r.x-v.y*r.y, v.x*r.y+v.y*r.x);
    }
    state[offset+a]=u; state[offset+b]=v;
}
extern "C" __global__ void probabilities(const float2* state, double* out, long size) {
    long i=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<size) { float2 z=state[i]; out[i]=(double)z.x*z.x+(double)z.y*z.y; }
}
extern "C" __global__ void walsh(double* x, int batch, int dim, int stride) {
    long k=(long)blockIdx.x*blockDim.x+threadIdx.x;
    if(k >= (long)batch*(dim/2)) return;
    int row=k/(dim/2), pair=k%(dim/2);
    long a=(long)row*dim+(pair/stride)*2*stride+pair%stride, b=a+stride;
    double u=x[a], v=x[b]; x[a]=u+v; x[b]=u-v;
}
'''


def suffixes(tokens, *, seed, g, gpu, microbatch=32):
    started = time.monotonic()
    n, t, d = tokens.shape
    q, dim = d//3, 1 << (d//3)
    cfg = DTCQRCConfig(num_qubits=q, g=float(g), W_scale=W, layers_per_injection=LAYERS)
    params = generate_dtc_params(q, seed, W)
    device = torch.device("cpu")
    phase = build_dtc_diagonal_phase(cfg, torch.from_numpy(params["J"]),
        torch.from_numpy(params["h"]), device, torch.complex64).numpy()
    drive = rx_gate(float(g)*np.pi, device, torch.complex64)
    values = np.ascontiguousarray(tokens, dtype=np.float32)*np.float32(base.ANGLE)
    raw = hamiltonian_modulation_gates(torch.from_numpy(values.reshape(-1, 3)), torch.complex64)
    fused = (drive @ raw).reshape(n, t, q, 2, 2).contiguous().numpy()
    c, s = np.float32(drive[0, 0].real), np.float32(-drive[0, 1].imag)
    masks = [1 << (q-1-i) for i in range(q)]
    masks += [(1 << (q-1-i)) ^ (1 << (q-1-j)) for i in range(q) for j in range(i+1, q)]
    output = np.empty((t, n, 3*len(masks)), np.float32)
    maximum_norm_error = 0.
    with cp.cuda.Device(gpu):
        kernels = cp.RawModule(code=CUDA, options=("--std=c++11", "--fmad=false"))
        gate, probabilities, walsh = [kernels.get_function(k) for k in ("gate", "probabilities", "walsh")]
        phase_d, masks_d = cp.asarray(phase), cp.asarray(masks, dtype=cp.int32)
        for start in range(0, n, microbatch):
            end = min(start+microbatch, n)
            count, total = end-start, (end-start)*t
            gates_d = cp.asarray(fused[start:end])
            state = cp.zeros((total, dim), cp.complex64)
            state[:, 0] = 1.

            def cycle(active, event=-1):
                size = active*(dim//2)
                for wire in range(q):
                    gate(((size+255)//256,), (256,), (state, gates_d, phase_d,
                        np.int32(active), np.int32(count), np.int32(t), np.int32(q), np.int32(dim),
                        np.int32(1 << (q-1-wire)), np.int32(event), np.int32(wire),
                        c, s, np.int32(wire == q-1)))

            for event in range(t):
                active = (event+1)*count
                cycle(active, event)
                for _ in range(1, LAYERS):
                    cycle(active)
            reduced = cp.empty((total, dim), cp.float64)
            for snapshot in range(3):
                if snapshot:
                    cycle(total)
                size = total*dim
                probabilities(((size+255)//256,), (256,), (state, reduced, np.int64(size)))
                for shift in range(q):
                    pairs = total*(dim//2)
                    walsh(((pairs+255)//256,), (256,),
                          (reduced, np.int32(total), np.int32(dim), np.int32(1 << shift)))
                maximum_norm_error = max(maximum_norm_error, float(cp.abs(reduced[:, 0]-1).max().get()))
                output[:, start:end, snapshot*len(masks):(snapshot+1)*len(masks)] = (
                    cp.asnumpy(reduced[:, masks_d]).reshape(t, count, len(masks)))
            del state, reduced, gates_d
        cp.cuda.get_current_stream().synchronize()
    assert maximum_norm_error < 5e-4
    assert np.isfinite(output).all() and np.abs(output).max() < 1.002
    return output, dict(maximum_norm_error=maximum_norm_error,
        elapsed_seconds=time.monotonic()-started, seed=seed, g=g, gpu=gpu,
        backend="CUDA-all-exact-suffixes-float32-state-float64-Walsh",
        params={k: params[k].tolist() for k in ("J", "h")})


def features(g, seed, gpu, microbatch=32, chunk_size=96):
    folder = folder_for(g, seed)
    output, marker = folder / "p020_features.npz", folder / "features_complete.json"
    if marker.exists():
        assert output.exists()
        return str(output)
    folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with np.load(base.case_dir(g, seed) / "features.npz", allow_pickle=False) as z:
        reference = z[pkey(0.)]
    tokens = dataset()["tokens"]
    result, records = np.empty((len(tokens), 360), np.float32), []
    for start in range(0, len(tokens), chunk_size):
        end = min(start+chunk_size, len(tokens))
        chunk = folder / "chunks" / f"rows{start:04d}_{end:04d}.npz"
        note = chunk.with_suffix(".json")
        if chunk.exists() and note.exists():
            record = json.loads(note.read_text())
            assert sha(chunk) == record["sha256"]
            with np.load(chunk, allow_pickle=False) as z:
                values = z["features"]
        else:
            bank, record = suffixes(tokens[start:end], seed=seed, g=g, gpu=gpu, microbatch=microbatch)
            error = float(np.abs(bank[0]-reference[start:end]).max())
            assert error < 5e-5, (g, seed, gpu, start, error)
            values = mix_suffix_bank(bank, .2, "tm3", 120)
            nwrite(chunk, features=values)
            record.update(start=start, end=end, closed_feature_error=error, sha256=sha(chunk))
            jwrite(note, record)
        result[start:end] = values
        records.append(record)
        jwrite(folder / "feature_progress.json", dict(g=g, seed=seed, gpu=gpu, rows_done=end,
            rows_total=len(tokens), elapsed_seconds=time.monotonic()-started))
        print(f"[GPU{gpu} features g={g:.2f} seed={seed}] {end}/{len(tokens)}", flush=True)
    assert np.isfinite(result).all() and np.abs(result).max() < 1.0002
    nwrite(output, features=result)
    jwrite(marker, dict(version=VERSION, g=g, seed=seed, p=.2, rows=2376, width=360,
        source="verified-CUDA-full-suffix-mixture-with-preserved-CPU-checkpoints", gpu=gpu,
        sha256=sha(output), bytes=output.stat().st_size, elapsed_seconds=time.monotonic()-started,
        maximum_closed_feature_error=max(r["closed_feature_error"] for r in records),
        maximum_norm_error=max(r["maximum_norm_error"] for r in records),
        chunk_backends=sorted(set(r.get("backend", "original-CPU") for r in records))))
    # Remove only this case's redundant chunks after verifying consolidation.
    assert sha(output) == json.loads(marker.read_text())["sha256"]
    for path in sorted((folder / "chunks").glob("rows*.npz")):
        path.unlink()
    for path in sorted((folder / "chunks").glob("rows*.json")):
        path.unlink()
    return str(output)
