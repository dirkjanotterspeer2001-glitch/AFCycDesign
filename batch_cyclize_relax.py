#!/usr/bin/env python3
"""
Batch cyclization + constrained relax for ProteinMPNN designs.

For each matching pair:
  PREFIX_{ID}.fa  (ProteinMPNN; designs with 'sample=')
  PREFIX_{ID}.pdb (backbone model; we restrict to chain A (or --chain) only)

This script:
  1) threads each 'sample=' sequence onto the target chain,
  2) cyclizes that chain with PeptideCyclizeMover (residue_selector schema),
  3) runs cartesian FastRelax with constraint terms ON and no ramp-down,
  4) re-declares the polymer bond (DeclareBond) and does a gentle touch-up,
  5) writes one output PDB per sequence (chain only).

Output naming (no cycles):
  {FASTA_prefix}_s{sequence_number}_{ID}.pdb

Where:
  - FASTA_prefix is derived from the .fa stem (everything before the last underscore)
  - ID is derived from the .fa/.pdb stem (substring after the last underscore)
  - sequence_number uses 'sample=' from header if present, otherwise i+1

Parallelization aids (two ledgers in --out_dir):
  - claimed_ids.txt   : IDs claimed for processing (prevents duplicates).
  - completed_ids.txt : IDs with ≥1 output PDB (human-readable completion log).
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Set, Iterable, Tuple, Optional

import pyrosetta
from pyrosetta import rosetta


# --------------------------- PyRosetta init ---------------------------

def init_pyrosetta():
    """
    Initialize PyRosetta for cartesian relax with constraints.
    """
    pyrosetta.init(
        " ".join([
            "-mute all",
            "-in:file:fullatom",
            "-ignore_unrecognized_res true",
            "-relax:cartesian",
            "-use_input_sc",
            "-flip_HNQ true",
            "-load_PDB_components false",
        ])
    )


# --------------------------- FASTA parsing ---------------------------

def parse_mpnn_fasta_entries(text: str):
    """
    Parse a FASTA (.fa) into a list of (header, seq), preserving order.
    """
    entries = []
    header = None
    seq = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None and seq:
                entries.append((header, "".join(seq)))
            header = line[1:]
            seq = []
        else:
            seq.append(line)
    if header is not None and seq:
        entries.append((header, "".join(seq)))
    return entries


def select_design_entries(entries):
    """
    Prefer the ProteinMPNN design entries (those whose header contains 'sample=').
    Fall back to all entries if none match.
    """
    subset = [(h, s) for (h, s) in entries if "sample=" in h]
    return subset if subset else entries


# --------------------------- Pose utilities ---------------------------

def get_chain_range(pose: rosetta.core.pose.Pose, chain_id: str):
    """
    Return (start, end, length) for the chain with PDB chain ID (e.g., 'A').
    """
    chain_index = rosetta.core.pose.get_chain_id_from_chain(chain_id, pose)
    start = pose.conformation().chain_begin(chain_index)
    end = pose.conformation().chain_end(chain_index)
    return start, end, end - start + 1


def thread_sequence(pose: rosetta.core.pose.Pose, chain_id: str, seq: str):
    """
    Thread 'seq' onto the specified chain by PDB chain ID.
    """
    start, _, length = get_chain_range(pose, chain_id)
    if len(seq) != length:
        raise ValueError(f"Sequence length {len(seq)} != chain {chain_id} length {length}")
    rosetta.protocols.simple_moves.SimpleThreadingMover(seq, start).apply(pose)


# --------------------------- RosettaScripts XML ---------------------------

def make_xml_relax_new(chain_id: str, relax_repeats: int) -> str:
    """
    Build XML that:
      - selects the target chain,
      - cyclizes it with PeptideCyclizeMover (residue_selector schema),
      - performs cartesian FastRelax with constraint terms ON, no ramp-down.
    """
    return f"""
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="relax_cst" weights="ref2015_cart">
      <Reweight scoretype="coordinate_constraint" weight="1.0"/>
      <Reweight scoretype="atom_pair_constraint"  weight="1.0"/>
      <Reweight scoretype="dihedral_constraint"   weight="1.0"/>
      <Reweight scoretype="angle_constraint"      weight="1.0"/>
      <Reweight scoretype="cart_bonded"           weight="0.5"/>
    </ScoreFunction>
  </SCOREFXNS>

  <RESIDUE_SELECTORS>
    <Chain name="sel_chain" chains="{chain_id}"/>
  </RESIDUE_SELECTORS>

  <MOVERS>
    <PeptideCyclizeMover name="cyclize" residue_selector="sel_chain"/>
    <FastRelax name="relax"
               scorefxn="relax_cst"
               cartesian="1"
               ramp_down_constraints="false"
               repeats="{relax_repeats}"/>
  </MOVERS>

  <PROTOCOLS>
    <Add mover="cyclize"/>
    <Add mover="relax"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
""".strip()


def make_xml_touchup_new(res1_end: int, res2_start: int) -> str:
    """
    Build XML that:
      - re-declares the terminal polymer bond (C@end -> N@start) to refresh O/H,
      - runs a gentle cartesian FastRelax with reduced constraint weights.
    """
    return f"""
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="touchup_cst" weights="ref2015_cart">
      <Reweight scoretype="coordinate_constraint" weight="0.5"/>
      <Reweight scoretype="atom_pair_constraint"  weight="0.5"/>
      <Reweight scoretype="dihedral_constraint"   weight="0.5"/>
      <Reweight scoretype="angle_constraint"      weight="0.5"/>
      <Reweight scoretype="cart_bonded"           weight="0.5"/>
    </ScoreFunction>
  </SCOREFXNS>

  <MOVERS>
    <DeclareBond name="update_polymer_bond"
                 atom1="C" res1="{res1_end}"
                 atom2="N" res2="{res2_start}"/>
    <FastRelax name="touchup_relax"
               scorefxn="touchup_cst"
               cartesian="1"
               ramp_down_constraints="true"
               repeats="2"/>
  </MOVERS>

  <PROTOCOLS>
    <Add mover="update_polymer_bond"/>
    <Add mover="touchup_relax"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
""".strip()


# --------------------------- I/O helpers ---------------------------

def sanitize_tag(s: str, maxlen: int = 64) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s[:maxlen] if len(s) > maxlen else s


def split_prefix_and_id(path: Path) -> Tuple[str, Optional[str]]:
    """
    Split a filename stem into (prefix, id) where id is the substring after the last underscore.
    If no underscore exists, returns (stem, None).

    Example: MC_target_0.fa -> ("MC_target", "0")
    """
    stem = path.stem
    if "_" not in stem:
        return stem, None
    prefix, rid = stem.rsplit("_", 1)
    return prefix, rid


def extract_id_from_name(path: Path) -> Optional[str]:
    """
    Generic ID extraction for files of the form PREFIX_{ID}.fa/.pdb:
    returns the substring after the last underscore in the stem.
    """
    _, rid = split_prefix_and_id(path)
    return rid


def extract_id_from_output(path: Path) -> Optional[str]:
    """
    Extract {ID} from output PDB name: {prefix}_s{n}_{ID}.pdb
    """
    m = re.match(r"^.+_s\d+_(.+)\.pdb$", path.name)
    return m.group(1) if m else None


def build_output_name(
    out_dir: Path,
    fasta_path: Path,
    run_id: str,
    sample_idx: int,
    header: str,
) -> Path:
    """
    Build output PDB path:
      {FASTA_prefix}_s{sequence_number}_{ID}.pdb

    - FASTA_prefix from the .fa filename (everything before last underscore).
    - sequence_number from header 'sample=' if present, else i+1.
    - ID is run_id (extracted from the .fa/.pdb filename).
    """
    fa_prefix, _ = split_prefix_and_id(fasta_path)

    m_samp = re.search(r"sample\s*=\s*([0-9]+)", header)
    samp = m_samp.group(1) if m_samp else str(sample_idx + 1)

    filename = f"{sanitize_tag(fa_prefix)}_s{samp}_{sanitize_tag(run_id)}.pdb"
    return out_dir / filename


# -------------------- Ledgers (claimed vs completed) --------------------

class FileLock:
    """
    Minimal cross-platform file lock (fcntl on POSIX, msvcrt on Windows, best-effort fallback).
    Used to serialize access to ledger files.
    """
    def __init__(self, file_obj):
        self._f = file_obj
        self._have_lock = False

    def __enter__(self):
        try:
            import fcntl
            fcntl.flock(self._f.fileno(), fcntl.LOCK_EX)
            self._have_lock = True
        except Exception:
            try:
                import msvcrt
                msvcrt.locking(self._f.fileno(), msvcrt.LK_LOCK, 1)
                self._have_lock = True
            except Exception:
                self._have_lock = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._have_lock:
            return
        try:
            import fcntl
            fcntl.flock(self._f.fileno(), fcntl.LOCK_UN)
        except Exception:
            try:
                import msvcrt
                msvcrt.locking(self._f.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass


def read_ids_file(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def write_ids_file(path: Path, ids: Iterable[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(sorted(set(ids))) + "\n")
    tmp.replace(path)


def append_id_locked(path: Path, rid: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    with path.open("r+", encoding="utf-8") as f, FileLock(f):
        f.seek(0)
        current = {line.strip() for line in f.readlines() if line.strip()}
        if rid in current:
            return
        f.seek(0, os.SEEK_END)
        f.write(rid + "\n")
        f.flush()
        os.fsync(f.fileno())


def scan_completed_ids(out_dir: Path) -> Set[str]:
    """
    Find IDs that already have output PDBs in out_dir (any sample).
    """
    done = set()
    for p in out_dir.glob("*_s*_*.pdb"):
        rid = extract_id_from_output(p)
        if rid:
            done.add(rid)
    return done


def refresh_ledgers_from_outputs(claimed_path: Path, completed_path: Path, out_dir: Path) -> Tuple[Set[str], Set[str]]:
    """
    Refresh both ledgers using current outputs.
      - completed := completed ∪ detected_from_outputs
      - claimed   := claimed ∪ completed
    Returns (claimed_set, completed_set).
    """
    existing_completed = read_ids_file(completed_path)
    detected = scan_completed_ids(out_dir)
    completed = existing_completed | detected
    write_ids_file(completed_path, completed)
    logging.info(
        f"Refreshed '{completed_path.name}': {len(completed)} completed "
        f"({len(detected)} detected from outputs)."
    )

    claimed = read_ids_file(claimed_path) | completed
    write_ids_file(claimed_path, claimed)
    logging.info(f"Refreshed '{claimed_path.name}': {len(claimed)} claimed (includes completed).")

    return claimed, completed


def claim_id(claimed_path: Path, completed_path: Path, run_id: str) -> bool:
    """
    Atomically claim an ID for processing:
      - lock claimed ledger
      - read claimed + (quick) completed; if present in either, return False
      - else append to claimed and return True
    """
    completed_now = read_ids_file(completed_path)
    if run_id in completed_now:
        return False

    claimed_path.parent.mkdir(parents=True, exist_ok=True)
    if not claimed_path.exists():
        claimed_path.touch()

    with claimed_path.open("r+", encoding="utf-8") as f, FileLock(f):
        f.seek(0)
        current = {line.strip() for line in f.readlines() if line.strip()}
        if run_id in current:
            return False
        f.seek(0, os.SEEK_END)
        f.write(run_id + "\n")
        f.flush()
        os.fsync(f.fileno())
        logging.info(f"Claimed ID '{run_id}'.")
        return True


def mark_completed(completed_path: Path, run_id: str):
    """
    Append run_id to the completed ledger (idempotent, locked).
    """
    append_id_locked(completed_path, run_id)
    logging.info(f"Marked ID '{run_id}' as completed.")


# --------------------------- Core processing ---------------------------

def process_one_run(
    pdb_path: Path,
    fasta_path: Path,
    out_dir: Path,
    chain_id: str,
    relax_repeats: int,
) -> int:
    """
    Process one {ID}: thread each design onto chain_id, cyclize+relax, write PDBs.
    We restrict the pose to the target chain (e.g. chain A),
    so only that chain is modeled and output.
    """
    # --- Load starting pose (full complex) ---
    full_pose = rosetta.core.import_pose.pose_from_file(str(pdb_path))

    # Locate the target chain in the full pose
    start_full, end_full, _ = get_chain_range(full_pose, chain_id)

    # --- Restrict to the target chain only ---
    pose = rosetta.core.pose.Pose(full_pose, start_full, end_full)

    # In this trimmed pose, the chain runs from 1..N
    start_idx = 1
    end_idx = pose.total_residue()
    chain_len = end_idx

    # Parse sequences (prefer the designs with 'sample=')
    entries = select_design_entries(parse_mpnn_fasta_entries(fasta_path.read_text()))
    if not entries:
        logging.warning(f"No sequences in {fasta_path.name}; skipping.")
        return 0

    # Build movers (parse XML once per run)
    xml_relax = rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        make_xml_relax_new(chain_id, relax_repeats)
    )
    relax_mvr = xml_relax.get_mover("ParsedProtocol")

    xml_touchup = rosetta.protocols.rosetta_scripts.XmlObjects.create_from_string(
        make_xml_touchup_new(end_idx, start_idx)  # C@end -> N@start in the trimmed pose
    )
    touch_mvr = xml_touchup.get_mover("ParsedProtocol")

    run_id = extract_id_from_name(pdb_path) or pdb_path.stem

    wrote = 0
    for i, (header, seq) in enumerate(entries):
        work = rosetta.core.pose.Pose()
        work.assign(pose)

        # Thread onto chain_id in this single-chain pose
        try:
            if len(seq) != chain_len:
                raise ValueError(f"design length {len(seq)} != chain {chain_id} length {chain_len}")
            thread_sequence(work, chain_id, seq)
        except Exception as e:
            logging.error(f"[{run_id}] skipping design {i+1}: {e}")
            continue

        # Cyclize + constrained relax
        relax_mvr.apply(work)

        # Re-declare polymer bond and light touch-up
        touch_mvr.apply(work)

        # Save
        out_path = build_output_name(out_dir, fasta_path, run_id, i, header)
        work.dump_pdb(str(out_path))
        wrote += 1
        logging.info(f"[{run_id}] wrote {out_path.name} (chain {chain_id} only)")

    return wrote


# --------------------------- CLI ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch cyclization + constrained relax for ProteinMPNN designs.")
    ap.add_argument("--fasta_dir", required=True, type=Path)
    ap.add_argument("--pdb_dir",   required=True, type=Path)
    ap.add_argument("--out_dir",   required=True, type=Path)
    ap.add_argument("--chain",     default="A", help="Target chain to thread/cyclize (default: A)")
    ap.add_argument("--relax_repeats", type=int, default=5)
    ap.add_argument("--glob", default="*.fa", help="Pattern for FASTA files (default: *.fa)")
    ap.add_argument("--claimed_ledger", type=str, default="claimed_ids.txt",
                    help="File name (inside --out_dir) tracking claimed IDs.")
    ap.add_argument("--completed_ledger", type=str, default="completed_ids.txt",
                    help="File name (inside --out_dir) tracking completed IDs.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    claimed_path = args.out_dir / args.claimed_ledger
    completed_path = args.out_dir / args.completed_ledger

    # 1) Refresh ledgers from current outputs
    refresh_ledgers_from_outputs(claimed_path, completed_path, args.out_dir)

    # 2) Initialize PyRosetta after quick I/O
    init_pyrosetta()

    # Build list of FASTA files
    fasta_files = sorted(args.fasta_dir.glob(args.glob))
    if not fasta_files:
        logging.error(f"No FASTA files found in {args.fasta_dir} matching {args.glob}")
        return

    # Map {ID} -> PDB
    pdb_map = {}
    for pdb in args.pdb_dir.glob("*.pdb"):
        rid = extract_id_from_name(pdb)
        if rid:
            pdb_map[rid] = pdb

    processed = 0
    skipped = 0

    # Snapshot of completed at start (skip quickly)
    completed_snapshot = read_ids_file(completed_path)

    for fa in fasta_files:
        rid = extract_id_from_name(fa)
        if not rid or rid not in pdb_map:
            logging.warning(f"Missing matching PDB for {fa.name}; skipping.")
            continue

        # Skip if already completed from snapshot
        if rid in completed_snapshot:
            logging.info(f"Skipping ID '{rid}' (already completed).")
            skipped += 1
            continue

        # Attempt to claim this ID for processing
        if not claim_id(claimed_path, completed_path, rid):
            logging.info(f"Skipping ID '{rid}' (already claimed/completed).")
            skipped += 1
            continue

        # Process the run
        n_written = 0
        try:
            n_written = process_one_run(
                pdb_map[rid],
                fa,
                args.out_dir,
                args.chain,
                args.relax_repeats,
            )
        except Exception as e:
            logging.exception(f"Error while processing ID '{rid}': {e}")

        # Mark completed if we produced outputs
        if n_written > 0:
            mark_completed(completed_path, rid)
            processed += 1
        else:
            logging.warning(f"ID '{rid}' produced no outputs; not marking as completed.")

    logging.info(f"Finished: {processed} processed to completion, {skipped} skipped.")
    logging.info(f"Claimed ledger   : {claimed_path}")
    logging.info(f"Completed ledger : {completed_path}")


if __name__ == "__main__":
    main()
