import os
import subprocess
import logging
from pathlib import Path
import argparse



def process(afcycdesign_environment, afcycdesign_script, params_dir, outdir_base, target_pdb, peptides_dir):
    # Logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)
    for pep_file in os.listdir(peptides_dir):
        if not pep_file.endswith(".pdb"):
            continue
        log.info(f"RUNNING PDB {pep_file}")

    model_dir = os.path.join(outdir_base, f"result_AFCycDesign")
    os.makedirs(model_dir, exist_ok=True)
    cmd = [
        afcycdesign_environment,
        afcycdesign_script,
        "binder",
        "--data_dir", params_dir,
        "--target", target_pdb,
        "--peptides_dir", peptides_dir,       # ❌ gebruik map, niet één PDB
        "--outdir", model_dir,
        "--cyclic",
        "--offset_type", "2",
    ]


    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # cwd=outdir_base
            cwd=model_dir
        )
        log.info(result.stdout)
        if result.stdout:
            log.info(result.stdout)
        if result.stderr:
            log.error(result.stderr)
    except subprocess.CalledProcessError as e:
        log.error(f"❌ afcycdesign failed for {pep_file}: {e}")

def main ():
    # ---- Config ----
    print("SCRIPT STARTED")
    ap = argparse.ArgumentParser()
    default_afcycdesign_script = "/home/s2528436/ColabDesign/af/examples/afdesign_tools_v2.py"
    default_afcycdesign_environment = "/home/s2528436/miniconda3/envs/colabdesign/bin/python"

    default_target_pdb = "/home/s2528436/prosculpt_afcycdesign/Examples/input_data/KIT.pdb"
    default_peptides_dir = "/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/KIT_test/01/2_mpnn/rosetta/afcyc_ready"
    default_outdir_base = "/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/KIT_test/01/3_afcycdesign"
    # default_target_pdb = "/home/s2528436/prosculpt_afcycdesign/Examples/input_data/DDR2.pdb"
    # default_peptides_dir = "/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_200/01/2_mpnn/rosetta/afcyc_ready"
    # default_outdir_base = "/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_200/01/3_afcycdesign"
    default_params_dir = "/home/s2528436/ColabDesign/af/examples/params"


    ap.add_argument("--afcycdesign_environment", type=Path, default=default_afcycdesign_environment)
    ap.add_argument("--afcycdesign_script", type=Path, default=default_afcycdesign_script)
    ap.add_argument("--params_dir", type=Path, default=default_params_dir)
    ap.add_argument("--out_dir", type=Path, default=default_outdir_base)
    ap.add_argument("--target_pdb", type=Path, default=default_target_pdb)
    ap.add_argument("--peptides_dir", type=Path, default=default_peptides_dir)
    args = ap.parse_args()
    afcycdesign_environment = args.afcycdesign_environment
    afcycdesign_script = args.afcycdesign_script
    params_dir = args.params_dir
    outdir_base = args.out_dir
    target_pdb = args.target_pdb
    peptides_dir = args.peptides_dir


    # Maak output map
    os.makedirs(outdir_base, exist_ok=True)
    process (afcycdesign_environment, afcycdesign_script, params_dir, outdir_base, target_pdb, peptides_dir)

if __name__ == "__main__":
    main()
