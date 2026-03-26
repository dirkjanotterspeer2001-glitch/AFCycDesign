#!/home/s2528436/prosculpt_afcycdesign/bin/python3
# -*- coding: utf-8 -*-

"""
Prosculpt pipeline driver (RfDiffusion → rechain → ProteinMPNN → Rosetta cyclize/relax → AF-CycDesign/AFdesign → postprocess)

NOTE:
- Updates:
  - cfg.pbd_path -> cfg.pdb_path
  - run_afcycdesign(): removed the meaningless loop and made batch call once (if you use --peptides_dir batch mode)
  - subprocess.run(): log return code and fail loudly if non-zero (optional but recommended)

This file expects your local modules:
  - prosculpt
  - process_afcycdesign
  - batch_cyclize_relax
and your Hydra config tree under ./config.
"""

from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
from hydra.core.hydra_config import HydraConfig

import os
import logging
import glob
import re
import shutil
import pathlib
import subprocess
import time
import random

from Bio import SeqIO
from Bio.PDB import PDBParser

import prosculpt
import process_afcycdesign
import batch_cyclize_relax


log = logging.getLogger(__name__)

scripts_folder = pathlib.Path(__file__).resolve().parent / "scripts"


# -----------------------
# Utilities
# -----------------------
def run_and_log(command, log_func=log.info, dry_run=False, cfg=None):
    """Runs a shell command using os.system and logs stdout/stderr status."""
    if log_func:
        log_func(command)
    if dry_run:
        return

    stat = os.system(command)
    wife = os.WIFEXITED(stat)
    exit_code = os.waitstatus_to_exitcode(stat)

    log.info(f"Command exited with status {stat} and WIFEXITED {wife}. Exit code: {exit_code}")
    if exit_code != 0:
        log.error(
            "There was an error running the command. We consider it fatal to prevent any file loss. "
            "Check the logs and contact the developer."
        )

        dodatek = ""
        if cfg is not None:
            auto_restart_count = cfg.get('auto_restart_count', 0)
            auto_restart = cfg.get('auto_restart', 0)
            log.info(f"auto_restart_count: {auto_restart_count}, auto_restart: {auto_restart}")
            if auto_restart_count < auto_restart:
                log.error("As per config, Prosculpt will be restarted automatically.")
                with open_dict(cfg):
                    cfg.auto_restart_count = auto_restart_count + 1
                dodatek = (
                    f"However, Prosculpt was autoRestarted {cfg.get('auto_restart_count', 0)} "
                    f"out of {cfg.get('auto_restart', 0)} times (as per config) after encountering the crash."
                )
                prosculptApp(cfg)

        raise Exception(f"Command exited with exit code {exit_code}\n\n{dodatek}")


def parse_additional_args(cfg, group):
    dodatni = ""
    for k, v in (cfg.get(group, {}) or {}).items():
        dodatni += f" {k} {v}"
    return dodatni


# -----------------------
# General config prep
# -----------------------
def general_config_prep(cfg):
    log.info("Running general_config_prep")
    os.makedirs(cfg.output_dir, exist_ok=True)

    with open_dict(cfg):
        cfg.output_dir = str(pathlib.Path(cfg.output_dir).resolve())
        cfg.pdb_path = str(pathlib.Path(cfg.pdb_path).resolve())
        cfg.pdb_path = os.path.abspath(cfg.pdb_path)

        log.info(f"output_dir is {cfg.output_dir}")
        log.info(f"pdb_path is {cfg.pdb_path}")

        cfg.rfdiff_out_dir = os.path.join(cfg.output_dir, "1_rfdiff")
        cfg.mpnn_out_dir = os.path.join(cfg.output_dir, "2_mpnn")
        cfg.afcycdesign_out_dir = os.path.join(cfg.output_dir, "3_afcycdesign")
        cfg.rfdiff_out_path = os.path.join(cfg.rfdiff_out_dir, "")

        cfg.path_for_parsed_chains = os.path.join(cfg.mpnn_out_dir, "parsed_pdbs.jsonl")
        cfg.path_for_assigned_chains = os.path.join(cfg.mpnn_out_dir, "assigned_pdbs.jsonl")
        cfg.path_for_fixed_positions = os.path.join(cfg.mpnn_out_dir, "fixed_pdbs.jsonl")
        cfg.path_for_tied_positions = os.path.join(cfg.mpnn_out_dir, "tied_pdbs.jsonl")

        cfg.fasta_dir = os.path.join(cfg.mpnn_out_dir, "seqs")
        cfg.rosetta_dir = os.path.join(cfg.mpnn_out_dir, "rosetta")
        cfg.rfdiff_pdb = os.path.join(cfg.rfdiff_out_path, "_0.pdb")
        cfg.afcyc_pdb_dir = os.path.join(cfg.rosetta_dir, "afcyc_ready")
        cfg.afcycdesign_result_dir = os.path.join(cfg.afcycdesign_out_dir, "result_AFCycDesign")

        if cfg.get("skipRfDiff", False):
            cfg.chains_to_design = " ".join(sorted({_[0] for _ in cfg.designable_residues}))
            log.info(f"Skipping RFdiff, redesigning only chains: {cfg.chains_to_design}")

        if "inference" not in cfg:
            cfg.inference = {}
        if "symmetry" not in cfg.inference:
            cfg.inference.symmetry = None

        if "omit_AAs" not in cfg:
            cfg.omit_AAs = "X"

        chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        if cfg.chains_to_design is None:
            if cfg.inference.symmetry is not None:
                breaks = int(str(cfg.inference.symmetry)[1:])
            else:
                breaks = cfg.contig.count("/0 ") + 1
            cfg.chains_to_design = " ".join(chain_letters[:breaks])
            log.info(f"Chains to design (by contig chain breaks): {cfg.chains_to_design}")

    for directory in [cfg.rfdiff_out_dir, cfg.mpnn_out_dir, cfg.afcycdesign_out_dir]:
        os.makedirs(directory, exist_ok=True)
        log.info(f"Made directory {directory}")


# -----------------------
# RfDiffusion
# -----------------------
def pass_config_to_rfdiff(cfg):
    """
    Create prosculpt2rfdiff.yaml containing only selected groups.
    """
    def keep_only_groups(pair):
        groups_to_pass = cfg.pass_to_rfdiff
        key, _value = pair
        return key in groups_to_pass

    new_cfg = dict(filter(keep_only_groups, cfg.items()))
    new_cfg["defaults"] = ["base", "_self_"]

    if "pdb_path" in cfg:
        if "inference" not in new_cfg:
            new_cfg["inference"] = {}
        with open_dict(new_cfg["inference"]):
            new_cfg["inference"]["input_pdb"] = cfg.pdb_path

    log.info(f"Saving this config for RfDiff:\n{OmegaConf.to_yaml(new_cfg)}")
    with open(os.path.join(cfg.output_dir, "prosculpt2rfdiff.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(new_cfg))


def run_rfdiff(cfg):
    """
    RFdiffusion generates new protein structures according to contig.
    """
    log.info("*************** Running run_rfdiff ***************")

    # Check if we already have outputs
    if len(glob.glob(os.path.join(cfg.rfdiff_out_path, "*.pdb"))) == cfg.num_designs_rfdiff:
        log.info(f"Found {cfg.num_designs_rfdiff} .pdb in {cfg.rfdiff_out_path}. Skipping RFdiff.")
        if len(glob.glob(os.path.join(cfg.rfdiff_out_path, "*.trb"))) != cfg.num_designs_rfdiff:
            log.critical(
                f"Found {len(glob.glob(os.path.join(cfg.rfdiff_out_path, '*.trb')))} trb files "
                f"in {cfg.rfdiff_out_path}, expected {cfg.num_designs_rfdiff}. "
                f"Likely crash during writing; remove orphan pdb and rerun."
            )
            raise Exception("Number of RfDiff .pdb files does not match RfDiff .trb files!")
        log.info("*************** Skipping RFdiffusion altogether ***************")
        return

    if r"{OUTPUT_PATH}" in cfg.rfdiff_run_command:
        rfdiff_run_command = (cfg.rfdiff_run_command).replace(r"{OUTPUT_PATH}", cfg.output_dir)
        out_path = "/output"
        rfdiff_out_path = "/output/1_rfdiff/"
    else:
        rfdiff_run_command = cfg.rfdiff_run_command
        rfdiff_out_path = cfg.rfdiff_out_path
        out_path = cfg.output_dir

    os.makedirs(cfg.output_dir + "/schedules", exist_ok=True)

    rfdiff_cmd_str = (
        f"{rfdiff_run_command} "
        f"inference.output_prefix={rfdiff_out_path} "
        f"'contigmap.contigs={cfg.contig}' "
        f"inference.num_designs={cfg.num_designs_rfdiff} "
        f"inference.cyclic=true "
        f"-cn prosculpt2rfdiff.yaml -cd {out_path}"
    )
    run_and_log(rfdiff_cmd_str, cfg=cfg)

    log.info("*************** After running RFdiffusion ***************")


def rechain_rfdiff_pdbs(cfg):
    """
    RFdiffusion often joins chains. Rechain IDs by detecting breaks by CA-CA distance.
    """
    log.info("Running rechain_rfdiff_pdbs")
    rf_pdbs = glob.glob(os.path.join(cfg.rfdiff_out_path, "*.pdb"))
    for pdb in rf_pdbs:
        run_and_log(
            f'{cfg.pymol_python_path} {scripts_folder / "rechain.py"} "{pdb}" "{pdb}" '
            f'--chain_break_cutoff_A {cfg.chain_break_cutoff_A}',
            cfg=cfg,
        )

        # Symmetry sanity: move disordered wrong-oligomer files away
        if cfg.inference.symmetry is not None:
            sym = str(cfg.inference.symmetry)
            if "tetrahedral" in sym:
                n_intended = 4
            elif "icosahedral" in sym:
                n_intended = 20
            elif "octahedral" in sym:
                n_intended = 8
            else:
                n_intended = int(sym[1:])

            n_actual = len([i for i in PDBParser().get_structure("rechained", pdb).get_chains()])
            if n_actual != n_intended:
                log.info(
                    f"intended N chains: {n_intended}, actual N: {n_actual}; moving to disordered/"
                )
                disordered_dir = os.path.join(cfg.rfdiff_out_path, "disordered")
                os.makedirs(disordered_dir, exist_ok=True)
                shutil.move(pdb, disordered_dir)

    log.info("After rechaining")


# -----------------------
# Checkpointing
# -----------------------
def save_checkpoint(folder, piece, value):
    with open(os.path.join(folder, f"checkpoint_{piece}.txt"), "w") as f:
        f.write(str(value))
        f.flush()
    log.info(f"Saving checkpoint {piece} = {value}")


def get_checkpoint(folder, piece, default=0):
    path = os.path.join(folder, f"checkpoint_{piece}.txt")
    if os.path.exists(path):
        with open(path, "r") as f:
            raw = f.read() or "0"
            value = int(raw)
            log.info(f"Reading checkpoint {piece}: {value}")
            return value
    log.info(f"Checkpoint {piece} doesn't exist. Returning default value {default}")
    return default


# -----------------------
# Rosetta → AF-CycDesign PDB conversion
# -----------------------
def convert_rosetta_to_afcycdesign(pdb_files, output_dir, new_chain="E"):
    """
    Convert Rosetta-style PDBs to AF-CycDesign-ready PDBs by changing chain ID only.
    """
    os.makedirs(output_dir, exist_ok=True)

    for pdb_path in pdb_files:
        basename = os.path.basename(pdb_path)
        new_pdb_path = os.path.join(output_dir, basename.replace(".pdb", f"_afcyc.pdb"))

        with open(pdb_path, "r") as f_in, open(new_pdb_path, "w") as f_out:
            for line in f_in:
                if line.startswith(("ATOM  ", "HETATM")):
                    new_line = line[:21] + new_chain + line[22:]
                    f_out.write(new_line)
                else:
                    f_out.write(line)

        print(f"[INFO] Converted {basename} → {new_pdb_path}")


# -----------------------
# Cycling: ProteinMPNN → (Rosetta cyclize/relax) → prepare peptides for AFdesign
# -----------------------
def do_cycling(cfg):
    """
    Loop: ProteinMPNN + Rosetta cyclize/relax (your current flow), repeated afcycdesign_mpnn_cycles times.
    """
    log.info("Running do_cycling")
    start_cycle = get_checkpoint(cfg.output_dir, "cycle", 0)
    content_status = get_checkpoint(cfg.output_dir, "content_status", 0)

    for cycle in range(start_cycle, cfg.afcycdesign_mpnn_cycles):
        log.info(f"=== Cycle {cycle} ===")

        trb_paths = None
        input_mpnn = cfg.rfdiff_out_dir

        if cycle != 0:
            cycle_directory = os.path.join(cfg.output_dir, "2_1_cycle_directory")

            if content_status == 1:
                if os.path.exists(cycle_directory):
                    shutil.rmtree(cycle_directory)
                os.makedirs(cycle_directory, exist_ok=True)
                save_checkpoint(cfg.output_dir, "content_status", 2)
                content_status = 2

            if content_status == 2:
                af_model_subdirs = glob.glob(os.path.join(cfg.afcycdesign_out_dir, "*"))
                log.info(f"afcycdesign model subdirs: {af_model_subdirs}")

                for model_subdir in af_model_subdirs:
                    af_pdbs = sorted(glob.glob(os.path.join(model_subdir, "T*.pdb")))
                    rf_model_num = prosculpt.get_token_value(os.path.basename(model_subdir), "model_", r"(\d+)")

                    number_of_copied = len([f for f in os.listdir(cycle_directory) if f"rf_{rf_model_num}" in f])
                    log.info(f"Already copied/present models: {number_of_copied}")

                    for i, af_pdb in enumerate(af_pdbs, start=number_of_copied):
                        af_model_num = prosculpt.get_token_value(os.path.basename(af_pdb), "model_", r"(\d+)")
                        shutil.move(
                            af_pdb,
                            os.path.join(
                                cycle_directory,
                                f"rf_{rf_model_num}__model_{af_model_num}__cycle_{cycle}__itr_{i}__.pdb",
                            ),
                        )

                save_checkpoint(cfg.output_dir, "content_status", 3)
                content_status = 3

            input_mpnn = cycle_directory

            if content_status != 4:
                shutil.rmtree(cfg.mpnn_out_dir)
                os.makedirs(cfg.mpnn_out_dir, exist_ok=True)

            trb_paths = os.path.join(cfg.rfdiff_out_dir, "*.trb")

        mpnn_run_command_only_path = (cfg.mpnn_run_command).rsplit("/", 1)[0]

        if content_status != 4:
            run_and_log(
                f"{mpnn_run_command_only_path}/helper_scripts/parse_multiple_chains.py "
                f"--input_path={input_mpnn} "
                f"--output_path={cfg.path_for_parsed_chains}",
                cfg=cfg,
            )

            if cfg.chains_to_design:
                run_and_log(
                    f"{mpnn_run_command_only_path}/helper_scripts/assign_fixed_chains.py "
                    f"--input_path={cfg.path_for_parsed_chains} "
                    f"--output_path={cfg.path_for_assigned_chains} "
                    f"--chain_list='{cfg.chains_to_design}'",
                    cfg=cfg,
                )

            # Creates fixed positions jsonl etc.
            _fixed_pos_path = prosculpt.process_pdb_files(
                input_mpnn, cfg.mpnn_out_dir, cfg, trb_paths, cycle=cycle
            )

            if cfg.inference.symmetry is not None:
                run_and_log(
                    f"{mpnn_run_command_only_path}/helper_scripts/make_tied_positions_dict.py "
                    f"--input_path={cfg.path_for_parsed_chains} "
                    f"--output_path={cfg.path_for_tied_positions} "
                    f"--homooligomer 1",
                    cfg=cfg,
                )

            proteinMPNN_cmd_str = (
                f"{cfg.mpnn_run_command} "
                f"--jsonl_path {cfg.path_for_parsed_chains} "
                f"--fixed_positions_jsonl {cfg.path_for_fixed_positions} "
                f"{('--tied_positions_jsonl ' + cfg.path_for_tied_positions) if cfg.inference.symmetry is not None else ''} "
                f"--chain_id_jsonl {cfg.path_for_assigned_chains} "
                f"--out_folder {cfg.mpnn_out_dir} "
                f"--num_seq_per_target {cfg.num_seq_per_target_mpnn if cycle == 0 else 1} "
                f"--sampling_temp {cfg.sampling_temp} "
                f"--backbone_noise {cfg.backbone_noise} "
                f"--use_soluble_model "
                f"--omit_AAs {cfg.omit_AAs} "
                f"{parse_additional_args(cfg, 'pass_to_mpnn')} "
                f"--batch_size 1"
            )
            run_and_log(proteinMPNN_cmd_str, cfg=cfg)

            # reset afcycdesign_out_dir for next stage
            shutil.rmtree(cfg.afcycdesign_out_dir)
            os.makedirs(cfg.afcycdesign_out_dir, exist_ok=True)

            # Make monomer fastas if symmetry or model_monomer
            fasta_files = sorted(glob.glob(os.path.join(cfg.fasta_dir, "*.fa")))
            monomers_fasta_dir = os.path.join(cfg.fasta_dir, "monomers")
            os.makedirs(monomers_fasta_dir, exist_ok=True)

            if cfg.inference.symmetry is not None or cfg.get("model_monomer", False):
                for fasta_file in fasta_files:
                    sequences = []
                    for record in SeqIO.parse(fasta_file, "fasta"):
                        record.seq = record.seq[: record.seq.find("/")]
                        rid = record.id
                        record.id = "monomer_" + rid
                        record.description = "monomer_" + record.description
                        sequences.append(record)
                    out_fa = os.path.join(
                        monomers_fasta_dir,
                        f"{os.path.basename(fasta_file)[:-3]}_monomer.fa",
                    )
                    SeqIO.write(sequences, out_fa, "fasta")
                    log.info(f"Wrote monomer fasta: {out_fa}")

        save_checkpoint(cfg.output_dir, "content_status", 4)
        content_status = 4

        # -----------------------
        # Rosetta cyclize/relax step
        # -----------------------
        os.makedirs(cfg.rosetta_dir, exist_ok=True)
        os.makedirs(cfg.fasta_dir, exist_ok=True)

        run_and_log(
            f"python3 batch_cyclize_relax.py "
            f"--fasta_dir {cfg.fasta_dir} "
            f"--pdb_dir {cfg.rfdiff_out_dir} "
            f"--out_dir {cfg.rosetta_dir} "
            f"--chain A "
            f"--relax_repeats {5}",
            cfg=cfg,
        )

        pdb_files = glob.glob(os.path.join(cfg.rosetta_dir, "*.pdb"))
        os.makedirs(cfg.afcyc_pdb_dir, exist_ok=True)
        convert_rosetta_to_afcycdesign(pdb_files, cfg.afcyc_pdb_dir)

        # ready for next cycle
        save_checkpoint(cfg.output_dir, "content_status", 1)
        save_checkpoint(cfg.output_dir, "cycle", cycle + 1)
        content_status = 1


# -----------------------
# AFdesign / AFCycDesign runner
# -----------------------
def run_afcycdesign(cfg):
    """
    Batch run AFcycDesign / AFDesign binder on a directory of peptide PDBs.
    """

    log.info("Start AFCycDesign/AFDesign")

    target_pdb = os.path.abspath(cfg.pdb_path)
    peptides_dir = cfg.afcyc_pdb_dir
    log.info("Peptides dir: %s", peptides_dir)
    afcycdesign_script = cfg.afcycdesign_run_command
    params_dir = cfg.afcycdesign_params_dir
    out_dir = os.path.abspath(cfg.afcycdesign_result_dir)
    log.info("Out_dir: %s", out_dir)



    # peptides_dir = "/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_10/01/2_mpnn/rosetta/afcyc_ready"
    # out_dir = "/home/s2528436/prosculpt_afcycdesign/Examples/Examples_out/DDR2_10/01/3_afcycdesign/result_AFCycDesign"
    os.makedirs(out_dir, exist_ok=True)





    cmd = [
        "singularity", "exec", "--nv",
        "--bind", f"{os.path.expanduser('~')}:{os.path.expanduser('~')}",
        "/home/s2528436/prosculpt_afcycdesign/colabdesign_latest.sif",
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
        "--idle_sec", "60",
        "--score", os.path.join(out_dir, "score.csv"),
    ]

    env = os.environ.copy()
    env["TMPDIR"] = out_dir
    env["TMP"] = out_dir
    env["TEMP"] = out_dir



    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=out_dir,
        env=env,
    )

    log.info(f"AFcycDesign return code: {result.returncode}")

    if result.stdout:
        log.info(result.stdout)

    if result.stderr:
        log.error(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"AFcycDesign failed with exit code {result.returncode}"
        )

def do_filtering(cfg, chain_A, chain_B):
    in_dir = cfg.afcycdesign_result_dir
    log.info(f"Inputmap from afcycdesign: {in_dir}")
    out_dir = os.path.join(cfg.afcycdesign_out_dir, "rosetta_filtered")
    os.makedirs(out_dir, exist_ok=True)
    out_prefix = os.path.join(out_dir, "rosetta_filtered")
    log.info(f"Output map: {out_dir}")
    cmd = [
        "python3",
        "/zfsstore/user/s2528436/prosculpt_afcycdesign/rosetta_filtering.py",
        "--in_dir", in_dir,
        "--out_prefix", out_prefix,
        "--chainA", chain_A,
        "--chainB", chain_B,
        "--extra_flags", "-corrections::beta_nov16 true",
    ]

    log.info(f"[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=out_dir,
    )
    log.info(f"AFdesign return code: {result.returncode}")
    if result.stdout:
        log.info(result.stdout)
    if result.stderr:
        log.error(result.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"AFdesign failed with exit code {result.returncode}")




# -----------------------
# Timing logger
# -----------------------
TIMEMEASURES = {}
TIMECALC = time.time()
PREVIOUS_MESSAGE = "Before app start"


def dtimelog(message, final=False):
    global TIMECALC, PREVIOUS_MESSAGE
    dt = round(time.time() - TIMECALC, 1)
    log.info(f"* * * {PREVIOUS_MESSAGE} lasted {dt} s. Running {message} * * *")
    TIMEMEASURES[PREVIOUS_MESSAGE] = dt
    PREVIOUS_MESSAGE = message
    TIMECALC = time.time()

    if final:
        for k, v in TIMEMEASURES.items():
            log.info(f"{k} lasted {v} s.")


# Crash test knobs (via Hydra CLI)
crash_at_error = 0
crash_at_cycle = 0


# -----------------------
# Main Hydra app
# -----------------------
@hydra.main(version_base=None, config_path="config", config_name="run")

def prosculptApp(cfg: DictConfig) -> None:
    log.info("Hydra starting")
    log.info("The following configuration was passed:\n" + OmegaConf.to_yaml(cfg))
    dtimelog("general_config_prep")
    general_config_prep(cfg)

    config = HydraConfig.get()
    config_name = config.job.config_name
    config_path = [p["path"] for p in config.runtime.config_sources if p["schema"] == "file"][-1]
    shutil.copy(os.path.join(config_path, config_name + ".yaml"), os.path.join(cfg.output_dir, "input.yaml"))

    global crash_at_error, crash_at_cycle
    crash_at_error = cfg.get("throw", -1)
    crash_at_cycle = cfg.get("crash_at_cycle", 0)

    if cfg.get("only_run_analysis", False):
        log.info("*** only_run_analysis set: skipping generation, running final ops only ***")
        return

    if not cfg.get("skipRfDiff", False):
        dtimelog("pass_config_to_rfdiff")
        pass_config_to_rfdiff(cfg)

        dtimelog("run_rfdiff")
        run_rfdiff(cfg)

        dtimelog("rechain_rfdiff_pdbs")
        rechain_rfdiff_pdbs(cfg)
    else:
        log.info("*** Skipping RfDiff ***")
        dtimelog("skipping RfDiff")
        shutil.copy(cfg.pdb_path, os.path.join(cfg.rfdiff_out_dir, "_0.pdb"))

    dtimelog("do_cycling")
    do_cycling(cfg)
    dtimelog("process_afcycdesign.process")
    run_afcycdesign(cfg)
    if cfg.get("use_rosetta_filtering", False):
        do_filtering(cfg, "A", "B")
        dtimelog("do_rosetta_filtering")
    dtimelog("Finished", True)


if __name__ == "__main__":
    print("File:", __file__)
    log.info(f"Before running prosculptApp, cwd = {os.getcwd()}")
    prosculptApp()
