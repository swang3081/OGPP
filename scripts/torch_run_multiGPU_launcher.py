import os, shlex, subprocess, threading, time, torch, sys

REPO_DIR   = "/home/swang3081/canonical_flow_matching/Voronoi-Flow-Matching/"
ENTRY_FILE = "scripts/run_uniGBN_1024_10k_linear_uniform_multipleGPU.py"
ARGS = [
    "--exp_name", "li_uni_8H100_1024_celebA_100MB_warmup_from_ckpt_40w_8kepoch_fixlr",
    "--data_path", "data/pts_celebA_1024_20w_sorted_01.npz",
    "--log_path",  "log",
    "--epochs","8241",
    "--lr","2e-6",
    "--batch_size","750",
    "--embed_dim","512",
    "--depth","8",
    "--num_heads","8",
    "--mlp_ratio","4.0",
    "--log_every","200",
    # "--data_aug_rotate",
    "--force_ddp",
    "--warmup_epochs", "200",
    # "--use_warmup",
    "--min_lr","1e-6",
    # "--use_cos_decay",
    "--if_load_ckpt",
    "--ckpt_path", "log/li_uni_8H100_1024_celebA_100MB_warmup_from_ckpt_40w_8kepoch_fixlr/checkpoints/UncondUniGBNTransformer_e8161_gs702280_lr1e-05_bs750_1764483238.pt",
    # "--only_load_model_weight"
]

NUM_PROCS = 2  # set to 1 for a single GPU now; change to 8 later for 8xH100

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1"

env.setdefault("MASTER_ADDR", "127.0.0.1")
env.setdefault("NCCL_IB_DISABLE", "1")
# env.setdefault("NCCL_SHM_DISABLE", "1")
env.pop("NCCL_SHM_DISABLE", None)           # allow NCCL to use shared memory
env.setdefault("OMP_NUM_THREADS", "1")
env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Optional: add the project root to PYTHONPATH, just to be safe
sep = os.pathsep
env["PYTHONPATH"] = REPO_DIR + (sep + env["PYTHONPATH"] if "PYTHONPATH" in env else "")

# Launch with the current interpreter to stay consistent with the notebook environment (fixes package mismatches such as numba)
py = sys.executable
cmd = [
    py, "-m", "torch.distributed.run",
    "--nproc-per-node", str(NUM_PROCS),
    "--rdzv_backend", "c10d",
    "--rdzv_endpoint", "127.0.0.1:29600",
    ENTRY_FILE, *ARGS
]
print(">> CMD:", " ".join(shlex.quote(c) for c in cmd), flush=True)

# Start the torchrun subprocess
proc = subprocess.Popen(
    cmd, cwd=REPO_DIR,
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    text=True, bufsize=1, env=env,
)

# === Print nvidia-smi every 60s (automatically exits when the subprocess ends) ===
def monitor_nvsmi(p, interval=60):
    while p.poll() is None:  # the subprocess is still running
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            print("\n---- nvidia-smi ----\n" + out.strip(), flush=True)
        except Exception as e:
            print(f"[nvsmi] error: {e}", flush=True)
        time.sleep(interval)

t = threading.Thread(target=monitor_nvsmi, args=(proc, 60), daemon=True)
t.start()

# Forward torchrun's logs in real time
for line in proc.stdout:
    print(line, end="", flush=True)

ret = proc.wait()
print(f"\n[torchrun exited with code {ret}]", flush=True)