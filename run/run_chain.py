# Run the downstream analysis chain after world_pipeline.py, with one log per stage.
# Stage groups run in order; the scripts inside a group run concurrently. A failed stage
# stops the chain and is reported with the tail of its log.
#   C:\Users\workt\anaconda3\python.exe run\run_chain.py            (all groups)
#   C:\Users\workt\anaconda3\python.exe run\run_chain.py figures    (figures and workbook only)
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOG = ROOT / "results_world" / "logs"
LOG.mkdir(exist_ok=True, parents=True)

GROUPS = {
    "classify": [["assign_koppen"], ["world_koppen_analysis"], ["dc_classification"]],
    "analyses": [["world_analysis", "closure_check", "gsa_world", "design_optimization",
                  "warming_scenarios", "aqueduct_overlay", "precip_phase",
                  "classification_hardening", "representativeness", "vcap_rederive",
                  "water_scenarios", "load_profiles", "approach_stats"],
                 ["stress_monthly_future"]],
    "figures": [["make_world_figs", "make_tier_figs", "make_ms_figs", "make_method_fig",
                 "make_si_fig3"], ["make_si_data", "build_master_workbook"]],
}


def run_group(names):
    procs = {}
    for name in names:
        log = open(LOG / f"{name}.log", "w")
        procs[name] = (subprocess.Popen([PY, str(ROOT / "run" / f"{name}.py")], stdout=log,
                                        stderr=subprocess.STDOUT, cwd=str(ROOT)), log)
    failed = []
    for name, (p, log) in procs.items():
        p.wait(); log.close()
        tail = (LOG / f"{name}.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()[-3:]
        print(f"  {name}: exit {p.returncode}  | " + " / ".join(tail), flush=True)
        if p.returncode != 0:
            failed.append(name)
    return failed


t0 = time.time()
which = sys.argv[1:] or list(GROUPS)
for g in which:
    for step in GROUPS[g]:
        print(f"[{g}] running {step} ({time.time()-t0:.0f}s)", flush=True)
        failed = run_group(step)
        if failed:
            print(f"[{g}] FAILED: {failed}; chain stopped", flush=True)
            sys.exit(1)
print(f"[chain done] {time.time()-t0:.0f}s", flush=True)
