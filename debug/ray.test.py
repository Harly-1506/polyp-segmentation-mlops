# import ray

# # Kết nối tới Ray cluster qua port-forward localhost
# ray.init(address="ray://127.0.0.1:10001")
# print("Cluster resources:", ray.cluster_resources())

# # Thử Ray task
# @ray.remote
# def f(x):
#     return x * x

# futures = [f.remote(i) for i in range(4)]
# print("Task results:", ray.get(futures))  # [0, 1, 4, 9]

# # Thử Ray actor
# @ray.remote
# class Counter:
#     def __init__(self):
#         self.n = 0

#     def increment(self):
#         self.n += 1

#     def read(self):
#         return self.n

# counters = [Counter.remote() for _ in range(4)]
# # Tăng giá trị mỗi actor
# [r.increment.remote() for r in counters]
# futures = [c.read.remote() for c in counters]
# print("Actor results:", ray.get(futures))  # [1, 1, 1, 1]
# import ray
# ray.init(address="ray://127.0.0.1:10001")

# @ray.remote
# def hello(x):
#     import torch
#     # print()
#     return f"Hello {x}, torch version: {torch.__version__}"


# print(ray.get(hello.remote("world")))

 # hoặc FQDN svc của bạn

# def node_key(n):
#     for k in n["Resources"]:
#         if k.startswith("node:") and k != "node:__internal_head__":
#             return k

# @ray.remote
# def probe():
#     import sys
#     out={"exe": sys.executable}
#     try:
#         import torch; out["torch"]=f"{torch.__version__} @ {getattr(torch,'__file__','')}"
#     except Exception as e:
#         out["torch"]=f"ERR: {e}"
#     return out

# for n in [n for n in ray.nodes() if n["Alive"]]:
#     k = node_key(n)
#     if not k: 
#         continue
#     res = ray.get(probe.options(resources={k:0.01}).remote())
#     print(k, "->", res)

# import json, os, sys, platform
# import ray
# from ray.exceptions import WorkerCrashedError

# # ----- Ray connect (ưu tiên RAY_ADDRESS, fallback localhost) -----
# RAY_ADDR = os.getenv("RAY_ADDRESS", "ray://127.0.0.1:10001")
# ray.init(address=RAY_ADDR, namespace=os.getenv("RAY_NAMESPACE", "default"),
#          ignore_reinit_error=True, log_to_driver=True)

# print("Ray version:", ray.__version__)
# print("cluster_resources:", ray.cluster_resources())
# print("available_resources:", ray.available_resources())

# def get_node_key(n: dict) -> str | None:
#     """Lấy resource key dạng node:<hostname> (không lấy __internal_head__)."""
#     for k in n.get("Resources", {}):
#         if k.startswith("node:") and k != "node:__internal_head__":
#             return k
#     return None

# # Task probe: không chiếm CPU để chạy được cả trên head (num-cpus=0)
# @ray.remote(max_retries=0, num_cpus=0)
# def import_only(modname: str):
#     import importlib, sys
#     m = importlib.import_module(modname)
#     return {
#         "mod": modname,
#         "version": getattr(m, "__version__", "?"),
#         "file": getattr(m, "__file__", "?"),
#         "exe": sys.executable,
#     }

# targets = ["numpy", "pyarrow", "torch"]
# nodes = [n for n in ray.nodes() if n.get("Alive")]

# results = {}
# for n in nodes:
#     nk = get_node_key(n)
#     if not nk:
#         # Bỏ qua node không có resource key (hiếm)
#         continue

#     results[nk] = {
#         "_node_ip": n.get("NodeManagerAddress"),
#         "_ray_node_name": n.get("NodeName"),
#         "_cpu_total": n.get("Resources", {}).get("CPU"),
#     }

#     for mod in targets:
#         try:
#             # Ghìm task vào đúng node bằng resource key; không yêu cầu CPU
#             out = ray.get(import_only.options(resources={nk: 0.01}).remote(mod))
#             results[nk][mod] = {"ok": True, **out}
#         except WorkerCrashedError:
#             results[nk][mod] = {"ok": False, "error": "WorkerCrashedError"}
#         except Exception as e:
#             results[nk][mod] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

# print(json.dumps(results, indent=2))

import json
import os
import platform
import socket
import sys

from ray.exceptions import GetTimeoutError, RayTaskError, WorkerCrashedError

import ray

# ----- Kết nối Ray (ưu tiên biến môi trường) -----
RAY_ADDR = "ray://kuberay-raycluster-head-svc.development:10001"

RAY_NS = os.getenv("RAY_NAMESPACE", "default")
ray.init(address=RAY_ADDR, namespace=RAY_NS, ignore_reinit_error=True, log_to_driver=True)

print("Ray version:", ray.__version__)
print("cluster_resources:", ray.cluster_resources())
print("available_resources:", ray.available_resources())

def node_resource_key(n: dict) -> str | None:
    """Trả về resource key 'node:...' đại diện cho node này (bỏ qua '__internal_head__')."""
    for k in n.get("Resources", {}):
        if k.startswith("node:") and k != "node:__internal_head__":
            return k
    return None

def is_head_node(n: dict) -> bool:
    """Xác định node có phải head bằng resource đặc biệt '__internal_head__'."""
    return "node:__internal_head__" in n.get("Resources", {})

# ---- Remote probe: ép chạy trên node mục tiêu (num_cpus=0 để không chiếm CPU) ----
@ray.remote(max_retries=0, num_cpus=0)
def node_probe(targets: list[str]) -> dict:
    import importlib
    import os
    import platform
    import socket
    import sys

    # Lấy CPU limit từ cgroup (cgroup v2 trước, fallback v1)
    def _cgroup_cpu_limit():
        try:
            # cgroup v2
            with open("/sys/fs/cgroup/cpu.max") as f:
                q, p = f.read().strip().split()
                if q == "max":
                    return None  # unlimited
                return float(q) / float(p)
        except Exception:
            try:
                # cgroup v1
                q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read().strip())
                p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read().strip())
                if q < 0:
                    return None
                return q / p
            except Exception:
                return None

    info = {
        "hostname": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "cgroup_cpu_limit": _cgroup_cpu_limit(),  # số CPU limit (float) hoặc None nếu unlimited
        "imports": {},
    }

    for mod in targets:
        try:
            m = importlib.import_module(mod)
            entry = {
                "ok": True,
                "version": getattr(m, "__version__", "?"),
                "file": getattr(m, "__file__", "?"),
            }
            if mod == "torch":
                import torch
                entry.update({
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_count": int(torch.cuda.device_count()) if hasattr(torch.cuda, "device_count") else 0,
                    "mean": float(torch.randn(8).mean().item()),
                })
            info["imports"][mod] = entry
        except Exception as e:
            info["imports"][mod] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    return info

targets = ["numpy", "pyarrow", "torch"]
nodes = [n for n in ray.nodes() if n.get("Alive")]

results: dict[str, dict] = {}

for n in nodes:
    nk = node_resource_key(n)
    if not nk:
        # Node không có resource key rõ ràng (hiếm), bỏ qua
        continue

    per_node = {
        "_node_ip": n.get("NodeManagerAddress"),
        "_ray_node_name": n.get("NodeName"),
        "_is_head": is_head_node(n),
        "_ray_resources": {k: v for k, v in n.get("Resources", {}).items() if not k.startswith("node:__internal_head__")},
    }

    try:
        # Ghìm task vào đúng node bằng resource key 'node:...' (dùng 0.001 để không xung đột)
        out = ray.get(
            node_probe.options(resources={nk: 0.001}).remote(targets),
            timeout=60,
        )
        per_node["probe"] = {"ok": True, **out}
    except (WorkerCrashedError, RayTaskError) as e:
        per_node["probe"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    except GetTimeoutError:
        per_node["probe"] = {"ok": False, "error": "GetTimeoutError: probe timeout"}
    except Exception as e:
        per_node["probe"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

    results[nk] = per_node

print(json.dumps(results, indent=2, ensure_ascii=False))

