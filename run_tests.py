import os
import subprocess
import logging

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Config test ---
afcycdesign_sif = "/home/s2528436/prosculpt_afcycdesign/colabdesign_latest.sif"
afcycdesign_script = "/home/s2528436/scripts/afdesign_tools_v3.py"
params_dir = "/home/s2528436/ColabDesign/af/examples/params"
target_pdb = "/home/s2528436/prosculpt_afcycdesign/Examples/input_data/DDR2.pdb"
peptides_dir = "/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_10/01/2_mpnn/rosetta/afcyc_ready"
out_dir = "/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_10/01/3_afcycdesign/result_AFCycDesign"
os.makedirs(out_dir, exist_ok=True)


# --- Build Singularity command ---
cmd = [
    "singularity", "exec", "--nv",
    "--bind", "/home/s2528436:/home/s2528436",
    afcycdesign_sif,
    "python3",
    afcycdesign_script,
    "binder",
    "--data_dir", params_dir,
    "--target", target_pdb,
    "--peptides_dir", peptides_dir,
    "--outdir", out_dir,
    "--cyclic",
    "--offset_type", "2",
    "--watch",
    "--idle_sec", "60",  # test met 60 sec
    "--score", os.path.join(out_dir, "score.csv"),
]

log.info(f"Running AFdesign test: {' '.join(cmd)}")

# --- Run command ---
result = subprocess.run(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    cwd=out_dir,
)

log.info(f"Return code: {result.returncode}")
if result.stdout:
    log.info(result.stdout)
if result.stderr:
    log.error(result.stderr)

if result.returncode != 0:
    raise RuntimeError(f"AFdesign test failed with exit code {result.returncode}")
else:
    log.info(f"AFdesign test completed successfully. Results in: {out_dir}")
