#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AFdesign utilities (combined):
- binder (batch): merge target (multi-chain) + macrocyclic peptide (single-chain, auto-assigned),
  quick multimer pass, save prediction + metrics, compute *aligned peptide RMSD* immediately,
  and automatically copy passing designs to a designated folder.
- fixbb        : fixed-backbone design on a given chain
- hallucination: de novo sequence generation for a given length

Key changes vs your posted script:
1) Multi-chain target support in binder mode:
   - Target chains are preserved (e.g., A,B or A,B,C).
   - Peptide is remapped to an automatically chosen unused chain ID (e.g., C or D).
   - binder_chain and target_chain passed to prep_inputs accordingly.
2) RMSD uses that same auto-assigned peptide chain ID (no need to specify --pred_chain).

Assumptions:
- colabdesign >= v1.1.x installed; AlphaFold params present
- Input PDBs already on disk
- Peptide input PDB contains only the peptide (any chain IDs inside will be remapped to the chosen peptide chain)

References (background):
- ColabDesign / AFdesign: Rettie et al., bioRxiv 2023 (doi:10.1101/2023.02.25.529956)
- AlphaFold-Multimer: Evans et al., bioRxiv 2021
"""

import os
import sys
import glob
import argparse
import numpy as np
import shutil
import time
import csv
import gzip
import string
import gc
import torch
import logging
from contextlib import contextmanager
from typing import List, Tuple, Optional

from colabdesign import mk_afdesign_model

# Biopython for RMSD alignment
from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.Polypeptide import is_aa

BACKBONE_ATOMS = ("N", "CA", "C", "O")
VALID_CHAIN_IDS = list(string.ascii_uppercase + string.ascii_lowercase + string.digits)


log = logging.getLogger(__name__)


# -----------------------
# PDB utilities (multi-chain target + auto peptide chain)
# -----------------------
def pdb_chain_ids_from_lines(pdb_lines: List[str]) -> List[str]:
    """Return chain IDs found in ATOM/HETATM records, in encounter order (unique)."""
    seen: List[str] = []
    seen_set = set()
    for ln in pdb_lines:
        if ln.startswith(("ATOM  ", "HETATM")):
            ch = ln[21]  # keep raw char (may be space)
            if ch not in seen_set:
                seen.append(ch)
                seen_set.add(ch)
    return seen

def choose_unused_chain_id(used_ids: List[str]) -> str:
    """
    Pick the first chain ID not in used_ids.
    - used_ids are single-character chain IDs, may include ' '.
    """
    used = set(used_ids)
    for cid in VALID_CHAIN_IDS:
        if cid not in used:
            return cid
    raise RuntimeError("No free chain IDs available to assign peptide chain.")

def _rewrite_atoms_keep_or_remap_chain(
    pdb_lines: List[str],
    keep_chains: Optional[Tuple[str, ...]] = None,
    remap_to: Optional[str] = None,
    atom_start: int = 1,
) -> Tuple[List[str], int]:
    """
    - keep_chains: if provided, keep only ATOM/HETATM lines whose chain ID is in keep_chains.
                  Chain ID is read as the raw single character at column 22 (index 21).
    - remap_to: if provided, overwrite the chain ID to this single-character chain.
    - atom_start: starting atom serial number
    """
    out: List[str] = []
    atom_idx = atom_start
    keep_set = set(keep_chains) if keep_chains is not None else None

    for ln in pdb_lines:
        if not ln.startswith(("ATOM  ", "HETATM")):
            continue

        ch = ln[21]
        if keep_set is not None and ch not in keep_set:
            continue

        # atom serial
        ln = f"{ln[:6]}{atom_idx:5d}{ln[11:]}"
        atom_idx += 1

        # chain remap
        if remap_to is not None:
            ln = ln[:21] + remap_to + ln[22:]

        out.append(ln)

    out.append("TER\n")
    return out, atom_idx

def merge_target_and_peptide_autochain(
    target_pdb: str,
    peptide_pdb: str,
    out_path: str,
    target_chains: Optional[Tuple[str, ...]] = None,  # None => keep ALL target chains found
) -> Tuple[str, str, Tuple[str, ...]]:
    """
    Multi-chain target merge:
    - Target chains are preserved.
    - Peptide atoms are remapped to an automatically chosen unused chain ID.
    Returns: (out_path, peptide_chain_id, target_chain_tuple)

    Notes:
    - Chain IDs are treated as single characters. Blank chain IDs (' ') are supported for merging,
      but using them in AFdesign chain selection may fail; see binder code for guard.
    """
    with open(target_pdb, "r") as f:
        t_lines = f.readlines()
    with open(peptide_pdb, "r") as f:
        p_lines = f.readlines()

    found_target_chains = pdb_chain_ids_from_lines(t_lines)
    if target_chains is None:
        use_target_chains = tuple(found_target_chains)
    else:
        use_target_chains = tuple(target_chains)

    if len(use_target_chains) == 0:
        raise RuntimeError("No target chains detected. Ensure target PDB has ATOM/HETATM records.")

    peptide_chain = choose_unused_chain_id(list(use_target_chains))

    merged: List[str] = []
    atom_idx = 1

    # Target: keep selected chains, preserve chain IDs
    block, atom_idx = _rewrite_atoms_keep_or_remap_chain(
        t_lines, keep_chains=use_target_chains, remap_to=None, atom_start=atom_idx
    )
    merged.extend(block)

    # Peptide: remap all atoms to peptide_chain
    block, atom_idx = _rewrite_atoms_keep_or_remap_chain(
        p_lines, keep_chains=None, remap_to=peptide_chain, atom_start=atom_idx
    )
    merged.extend(block)

    merged.append("END\n")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(merged)

    return out_path, peptide_chain, use_target_chains

# -----------------------
# Cyclic offset helpers
# -----------------------
def _cyclic_offset_matrix(L, offset_type=2):
    i = np.arange(L)
    ij = np.stack([i, i + L], -1)
    offset = np.array(i[:, None] - i[None, :])
    c_off = np.abs(ij[:, None, :, None] - ij[None, :, None, :]).min((2, 3))
    if offset_type >= 2:
        a = c_off < np.abs(offset)
        c_off[a] = -c_off[a]
    if offset_type == 3:
        idx = np.abs(c_off) > 2
        c_off[idx] = (32 * c_off[idx]) / np.abs(c_off[idx])
    return c_off * np.sign(offset)

def add_cyclic_offset(model, offset_type=2):
    """
    Adds cyclic offset to connect N–C for:
      - binder: binder block only (assumes binder is one contiguous designed block; fits your constraint)
      - fixbb / hallucination: each designed block
    """
    idx = model._inputs["residue_index"]
    offset = np.array(idx[:, None] - idx[None, :])

    if model.protocol == "binder":
        tL = model._target_len
        bL = model._binder_len
        offset[tL:tL+bL, tL:tL+bL] = _cyclic_offset_matrix(bL, offset_type)
    elif model.protocol in ["fixbb", "partial", "hallucination"]:
        Ln = 0
        for L in model._lengths:
            offset[Ln:Ln+L, Ln:Ln+L] = _cyclic_offset_matrix(L, offset_type)
            Ln += L
    model._inputs["offset"] = offset

# -----------------------
# Locking / claiming IDs
# -----------------------
@contextmanager
def claim_id(id_str, lock_dir, stale_sec=None):
    """
    Attempt to 'claim' an ID by creating a per-ID lock file atomically.
    Returns a context that yields the lock path if acquired, or None if not.

    - lock_dir: shared directory for lock files (must be on a shared filesystem).
    - stale_sec: if provided, an existing lock older than this many seconds is
      considered stale and will be removed before attempting to claim.

    The lock is released (file removed) when leaving the context, even on error.
    """
    os.makedirs(lock_dir, exist_ok=True)
    lock_path = os.path.join(lock_dir, f"{id_str}.lock")
    fd = None

    if os.path.exists(lock_path) and stale_sec is not None:
        try:
            age = time.time() - os.path.getmtime(lock_path)
            if age > stale_sec:
                os.remove(lock_path)
        except FileNotFoundError:
            pass

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, f"pid={os.getpid()}\n".encode("utf-8"))
        except Exception:
            pass
        yield lock_path
    except FileExistsError:
        yield None
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except Exception:
                pass
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

# -----------------------
# Correct RMSD utilities (Bio.PDB)
# -----------------------
def open_text_maybe_gz(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "rt")

def load_model(pdb_path: str):
    parser = PDBParser(QUIET=True)
    with open_text_maybe_gz(pdb_path) as handle:
        structure = parser.get_structure(os.path.basename(pdb_path), handle)
    return next(structure.get_models())

def chain_ids(model) -> List[str]:
    return [c.id for c in model.get_chains()]

def first_chain_id(model) -> Optional[str]:
    ids = chain_ids(model)
    return ids[0] if ids else None

def load_chain_atoms(model, chain_id: str, atom_names: Tuple[str, ...]) -> List:
    chain = model[chain_id]
    atoms: List = []
    for res in chain.get_residues():
        if not is_aa(res, standard=True):
            continue
        for aname in atom_names:
            if aname in res:
                atoms.append(res[aname])
    return atoms

def match_atoms_by_order(ref_atoms: List, mob_atoms: List) -> Tuple[List, List]:
    n = min(len(ref_atoms), len(mob_atoms))
    if n < 3:
        return [], []
    return ref_atoms[:n], mob_atoms[:n]

def aligned_peptide_rmsd(
    ref_pdb: str,
    pred_pdb: str,
    ref_chain: Optional[str] = None,
    preferred_pred_chain: str = "B",
    ca_only: bool = True,
) -> Optional[float]:
    """
    RMSD after superposition of predicted peptide chain onto the reference peptide.
    - ref_chain: if None, uses the first chain found in the reference PDB.
    - preferred_pred_chain: try this chain first in prediction PDB; if absent or fails,
      try all chains and take best.
    - ca_only: if True, use CA atoms only; else use backbone atoms (N,CA,C,O).
    """
    atom_names = ("CA",) if ca_only else BACKBONE_ATOMS

    try:
        ref_model = load_model(ref_pdb)
        pred_model = load_model(pred_pdb)
    except Exception:
        return None

    if ref_chain is None:
        ref_chain = first_chain_id(ref_model)
        if ref_chain is None:
            return None

    try:
        ref_atoms = load_chain_atoms(ref_model, ref_chain, atom_names)
    except Exception:
        return None
    if len(ref_atoms) < 3:
        return None

    pred_chains = chain_ids(pred_model)

    def rmsd_for_pred_chain(pc: str) -> Optional[float]:
        try:
            mob_atoms = load_chain_atoms(pred_model, pc, atom_names)
        except Exception:
            return None
        ref_m, mob_m = match_atoms_by_order(ref_atoms, mob_atoms)
        if len(ref_m) < 3:
            return None
        sup = Superimposer()
        sup.set_atoms(ref_m, mob_m)
        return float(sup.rms)

    if preferred_pred_chain in pred_chains:
        rmsd = rmsd_for_pred_chain(preferred_pred_chain)
        if rmsd is not None:
            return rmsd

    best_rmsd = None
    for pc in pred_chains:
        rmsd = rmsd_for_pred_chain(pc)
        if rmsd is None:
            continue
        if best_rmsd is None or rmsd < best_rmsd:
            best_rmsd = rmsd
    return best_rmsd

# -----------------------
# Binder (batch) mode
# -----------------------
def _list_peptide_pdbs(peptides_dir):
    return sorted(glob.glob(os.path.join(peptides_dir, "*.pdb")))

def _ready_files(candidates, probe_delay):
    """Return files whose size is stable across two probes."""
    sizes1 = {p: os.path.getsize(p) for p in candidates if os.path.isfile(p)}
    time.sleep(max(0.1, probe_delay))
    ready = []
    for p in candidates:
        if not os.path.isfile(p):
            continue
        s2 = os.path.getsize(p)
        if p in sizes1 and s2 == sizes1[p]:
            ready.append(p)
    return ready

def _load_processed_from_score(score_file):
    done = set()
    if os.path.exists(score_file):
        with open(score_file, newline="") as fh:
            r = csv.reader(fh)
            next(r, None)  # header
            for row in r:
                if row and row[0]:
                    done.add(row[0])
    return done

def _init_csv_if_missing(path: str, header: List[str]) -> None:
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)

def run_binder_batch(
    target_pdb, peptides_dir, outdir,
    score_file, single_peptide=None,
    multimer=True, initial_guess=True,
    cyclic=True, offset_type=2,
    data_dir=None, keep_temp=False,
    watch=False, poll_sec=60, idle_sec=900,
    lock_dir=None, lock_stale_sec=86400,
    # RMSD + pass/copy controls
    ipaethr=0.4, rmsdthr=1.5,
    passed_dir=None,
    ca_only=True,
    ref_chain=None,
    passed_csv_name="passed_scores_RMSD.csv",
    # NEW: optionally restrict which target chains to keep (comma-separated, e.g. "A,B" or "A,B,C")
    # If None: keep all chains present in the target PDB.
    target_chains_csv=None,
):
    os.makedirs(outdir, exist_ok=True)
    temp_dir = os.path.join(outdir, "temp")
    os.makedirs(temp_dir, exist_ok=True)

    local_lock_dir = lock_dir or os.path.join(outdir, ".locks")
    os.makedirs(local_lock_dir, exist_ok=True)

    # Main score file (all attempted peptides)
    if not os.path.exists(score_file):
        with open(score_file, "w") as f:
            f.write("peptide,iPAE,RMSD_aligned,pass,pred_pdb,peptide_chain,target_chains\n")

    # Passed folder
    if passed_dir is None:
        passed_dir = os.path.join(outdir, "Passed")
    os.makedirs(passed_dir, exist_ok=True)

    passed_csv = os.path.join(passed_dir, passed_csv_name)
    _init_csv_if_missing(passed_csv, ["peptide", "iPAE", "RMSD", "peptide_chain", "target_chains"])

    processed = _load_processed_from_score(score_file)

    # Parse optional target chain restriction
    target_chains_tuple = None
    if target_chains_csv:
        # Accept "A,B" or "A,B,C"
        target_chains_tuple = tuple([x.strip() for x in target_chains_csv.split(",") if x.strip()])
        if len(target_chains_tuple) == 0:
            target_chains_tuple = None

    start_no_new = None
    try:
        while True:
            processed = _load_processed_from_score(score_file)
            if single_peptide:
                log.info(f"[INFO] single_peptide mode ON: {single_peptide}")
            else:
                log.info("[INFO] single_peptide mode OFF: processing all peptides in directory")

            if single_peptide:
                if single_peptide in processed:
                    ready = []
                else:
                    ready = [single_peptide]
            else:
                all_pdbs = _list_peptide_pdbs(peptides_dir)
                todo = [p for p in all_pdbs if os.path.splitext(os.path.basename(p))[0] not in processed]
                ready = _ready_files(todo, probe_delay=max(1, poll_sec/2))

            if ready:
                start_no_new = None
                for pep in ready:
                    pep_base = os.path.splitext(os.path.basename(pep))[0]

                    if pep_base in _load_processed_from_score(score_file):
                        processed.add(pep_base)
                        continue

                    with claim_id(pep_base, local_lock_dir, stale_sec=lock_stale_sec) as lock:
                        if lock is None:
                            print(f"[SKIP] {pep_base}: claimed by another worker")
                            continue

                        if pep_base in _load_processed_from_score(score_file):
                            processed.add(pep_base)
                            print(f"[SKIP] {pep_base}: already done after lock acquisition")
                            continue

                        tgt_base = os.path.splitext(os.path.basename(target_pdb))[0]
                        combo_base = f"{pep_base}_on_{tgt_base}"
                        combo_pdb  = os.path.join(temp_dir, f"{combo_base}.pdb")
                        pred_pdb   = os.path.join(outdir, f"{combo_base}_prediction.pdb")

                        ipae = None
                        rmsd_aln = None
                        passed = False
                        pep_chain = None
                        tgt_chains = None

                        try:
                            # Merge: preserve multi-chain target; auto-assign peptide chain
                            combo_pdb, pep_chain, tgt_chains = merge_target_and_peptide_autochain(
                                target_pdb=target_pdb,
                                peptide_pdb=pep,
                                out_path=combo_pdb,
                                target_chains=target_chains_tuple
                            )

                            # Build target_chain argument for AFdesign
                            # Guard: blank chain IDs (' ') are difficult to select via chain strings.
                            if any(c == " " for c in tgt_chains):
                                raise RuntimeError(
                                    "Target PDB includes blank chain ID (' '). Please assign explicit chain IDs "
                                    "(e.g., A,B,...) before running binder mode."
                                )
                            target_chain_arg = ",".join(tgt_chains)

                            model = mk_afdesign_model("binder", data_dir=data_dir) if data_dir else mk_afdesign_model("binder")
                            model.prep_inputs(
                                combo_pdb,
                                binder_chain=pep_chain,
                                target_chain=target_chain_arg,
                                use_binder_template=False,
                                use_multimer=multimer,
                                use_initial_guess=initial_guess
                            )

                            if cyclic:
                                add_cyclic_offset(model, offset_type=offset_type)

                            model.set_seq(mode="wildtype")
                            model.set_opt(num_recycles=1)
                            model.predict(verbose=False)
                            model.save_pdb(pred_pdb)

                            # iPAE from AFdesign aux (fallbacks)
                            try:
                                x = model.aux["all"]["losses"]["i_pae"]
                                ipae = float(np.ravel(x)[0])
                            except Exception:
                                try:
                                    x = model.aux["losses"]["i_pae"]
                                    ipae = float(np.ravel(x)[0])
                                except Exception:
                                    ipae = None

                            # Aligned RMSD vs reference peptide structure (pep)
                            # Use auto-assigned peptide chain in prediction as preferred.
                            rmsd_aln = aligned_peptide_rmsd(
                                ref_pdb=pep,
                                pred_pdb=pred_pdb,
                                ref_chain=ref_chain,
                                preferred_pred_chain=pep_chain,
                                ca_only=ca_only,
                            )

                            if (ipae is not None) and (rmsd_aln is not None):
                                passed = (ipae < ipaethr) and (rmsd_aln <= rmsdthr)

                            tgt_chain_str = ",".join(tgt_chains) if tgt_chains else ""
                            pep_chain_str = pep_chain if pep_chain else ""

                            # Log into main score file (always)
                            with open(score_file, "a", newline="") as f:
                                w = csv.writer(f)
                                w.writerow([pep_base, ipae, rmsd_aln, int(passed), pred_pdb, pep_chain_str, tgt_chain_str])

                            if passed:
                                try:
                                    shutil.copy2(pred_pdb, os.path.join(passed_dir, os.path.basename(pred_pdb)))
                                except Exception as ce:
                                    print(f"[WARN] Copy failed for {pep_base}: {ce}")

                                with open(passed_csv, "a", newline="") as f:
                                    w = csv.DictWriter(f, fieldnames=["peptide", "iPAE", "RMSD", "peptide_chain", "target_chains"])
                                    w.writerow({
                                        "peptide": pep_base,
                                        "iPAE": f"{ipae:.6f}" if ipae is not None else "",
                                        "RMSD": f"{rmsd_aln:.4f}" if rmsd_aln is not None else "",
                                        "peptide_chain": pep_chain_str,
                                        "target_chains": tgt_chain_str,
                                    })
                                print(f"[PASS] {pep_base} | pep_chain={pep_chain_str} target={tgt_chain_str} | iPAE={ipae:.6f} RMSD={rmsd_aln:.4f} → copied to {passed_dir}")
                            else:
                                print(f"[OK] {pep_base} | pep_chain={pep_chain_str} target={tgt_chain_str} | iPAE={ipae} RMSD={rmsd_aln} → not passing filters")

                            processed.add(pep_base)

                        except Exception as e:
                            print(f"[FAIL] {pep_base}: {e}")
                            tgt_chain_str = ",".join(tgt_chains) if tgt_chains else ""
                            pep_chain_str = pep_chain if pep_chain else ""
                            with open(score_file, "a") as f:
                                f.write(f"{pep_base},{ipae},{rmsd_aln},0,{pred_pdb},{pep_chain_str},{tgt_chain_str}\n")
                            processed.add(pep_base)

                        finally:
                            # **Memory cleanup**
                            model._inputs = None
                            model.aux = None
                            del model
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()



            else:
                if not watch:
                    break

                now = time.time()
                elapsed_idle = 0 if start_no_new is None else now - start_no_new
                if start_no_new is None:
                    start_no_new = now
                    elapsed_idle = 0

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Waiting for new peptide PDBs... "
                      f"Processed so far: {len(processed)} | "
                      f"Idle for {int(elapsed_idle)}s (stop after {idle_sec}s)")
                sys.stdout.flush()

                if elapsed_idle >= idle_sec:
                    print(f"[INFO] Idle for {idle_sec}s; exiting watch mode.")
                    break

                time.sleep(poll_sec)

    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"[CLEANUP] Removed temporary folder: {temp_dir}")

# -----------------------
# Fixbb mode (single)
# -----------------------
def run_fixbb(
    pdb_path, chain="A", out_prefix="fixbb",
    cyclic=False, offset_type=2,
    design_steps=(50, 50, 10),
    data_dir=None
):
    model = mk_afdesign_model("fixbb", data_dir=data_dir) if data_dir else mk_afdesign_model("fixbb")
    model.prep_inputs(pdb_filename=pdb_path, chain=chain)
    if cyclic:
        add_cyclic_offset(model, offset_type=offset_type)

    model.restart()
    model.design_3stage(*design_steps)
    out_pdb = f"{out_prefix}.pdb"
    model.save_pdb(out_pdb)
    print(f"[OK] fixbb saved: {out_pdb}")

    try:
        seqs = model.get_seqs()
        with open(f"{out_prefix}.fa", "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">{out_prefix}_{i}\n{s}\n")
        print(f"[OK] sequences: {out_prefix}.fa")
    except Exception:
        pass

# -----------------------
# Hallucination mode
# -----------------------
def run_hallucination(
    length=50, rm_aa="C", out_prefix="hallucination",
    cyclic=False, offset_type=2,
    pre_soft_iters=50, stage_iters=(50, 50, 10),
    data_dir=None
):
    model = mk_afdesign_model("hallucination", data_dir=data_dir) if data_dir else mk_afdesign_model("hallucination")
    model.prep_inputs(length=length, rm_aa=rm_aa)
    if cyclic:
        add_cyclic_offset(model, offset_type=offset_type)

    model.restart()
    model.set_seq(mode="gumbel")
    model.set_opt("con", binary=True, cutoff=21.6875, num=length, seqsep=0)
    model.set_weights(pae=1, plddt=1, con=0.5)
    model.design_soft(pre_soft_iters)

    model.set_seq(seq=model.aux["seq"]["pseudo"])
    model.design_3stage(*stage_iters)

    out_pdb = f"{out_prefix}.pdb"
    model.save_pdb(out_pdb)
    print(f"[OK] hallucination saved: {out_pdb}")

    try:
        seqs = model.get_seqs()
        with open(f"{out_prefix}.fa", "w") as f:
            for i, s in enumerate(seqs):
                f.write(f">{out_prefix}_{i}\n{s}\n")
        print(f"[OK] sequences: {out_prefix}.fa")
    except Exception:
        pass

# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser(description="AFdesign quick tools (binder batch with aligned RMSD, fixbb, hallucination)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("binder", help="Batch binder quick pass for peptides vs multi-chain target (auto peptide chain)")
    pb.add_argument("--target", required=True)
    pb.add_argument("--peptides_dir", required=True)
    pb.add_argument("--outdir", default="predictions")
    pb.add_argument("--score", default="score.sc")
    pb.add_argument("--single_peptide", default=None,
                help="Run only a single peptide (used internally for subprocess mode)")
    pb.add_argument("--no-multimer", action="store_true")
    pb.add_argument("--no-initial-guess", action="store_true")
    pb.add_argument("--cyclic", action="store_true")
    pb.add_argument("--offset_type", type=int, default=2, choices=[1, 2, 3])
    pb.add_argument("--data_dir", default=None, help="Path to AlphaFold params")
    pb.add_argument("--keep_temp", action="store_true",
                    help="Keep merged target+peptide temp files in <outdir>/temp")

    pb.add_argument("--watch", action="store_true",
                    help="Continuously watch peptides_dir for new *.pdb and process as they appear")
    pb.add_argument("--poll_sec", type=int, default=60,
                    help="Polling interval when --watch is set (default 60s)")
    pb.add_argument("--idle_sec", type=int, default=600,
                    help="Stop after no new ready files for this long (default 600s)")

    pb.add_argument("--lock_dir", default=None,
                    help="Directory for lock files (default: <outdir>/.locks)")
    pb.add_argument("--lock_stale_sec", type=int, default=86400,
                    help="Reclaim locks older than this many seconds (default 86400s)")

    # filters + copying
    pb.add_argument("--ipaethr", type=float, default=0.4, help="Pass threshold: iPAE < ipaethr")
    pb.add_argument("--rmsdthr", type=float, default=1.5, help="Pass threshold: aligned RMSD <= rmsdthr")
    pb.add_argument("--passed_dir", default=None, help="Where to copy passing prediction PDBs (default: <outdir>/Passed)")
    pb.add_argument("--passed_csv_name", default="passed_scores_RMSD.csv",
                    help="Filename for passed CSV inside passed_dir")

    pb.add_argument("--ca_only", action="store_true", help="Use CA atoms only for RMSD (default: backbone atoms)")
    pb.add_argument("--ref_chain", default=None,
                    help="Reference chain ID in peptide PDB (default: auto-detect first chain)")

    # NEW: optional target chain restriction (if you want only A,B from a larger target)
    pb.add_argument("--target_chains", default=None,
                    help="Comma-separated target chains to keep (e.g., A,B or A,B,C). Default: keep all target chains.")

    pf = sub.add_parser("fixbb", help="Fixed-backbone design")
    pf.add_argument("--pdb", required=True)
    pf.add_argument("--chain", default="A")
    pf.add_argument("--out_prefix", default="fixbb")
    pf.add_argument("--cyclic", action="store_true")
    pf.add_argument("--offset_type", type=int, default=2, choices=[1, 2, 3])
    pf.add_argument("--stage_iters", type=int, nargs=3, default=[50, 50, 10])
    pf.add_argument("--data_dir", default=None, help="Path to AlphaFold params")

    ph = sub.add_parser("hallucination", help="De novo sequence generation")
    ph.add_argument("--length", type=int, required=True)
    ph.add_argument("--rm_aa", default="C")
    ph.add_argument("--out_prefix", default="hallucination")
    ph.add_argument("--cyclic", action="store_true")
    ph.add_argument("--offset_type", type=int, default=2, choices=[1, 2, 3])
    ph.add_argument("--pre_soft_iters", type=int, default=50)
    ph.add_argument("--stage_iters", type=int, nargs=3, default=[50, 50, 10])
    ph.add_argument("--data_dir", default=None, help="Path to AlphaFold params")

    args = p.parse_args()

    if args.cmd == "binder":
        run_binder_batch(
            target_pdb=args.target,
            peptides_dir=args.peptides_dir,
            outdir=args.outdir,
            single_peptide=args.single_peptide,
            score_file=args.score,
            multimer=(not args.no_multimer),
            initial_guess=(not args.no_initial_guess),
            cyclic=args.cyclic,
            offset_type=args.offset_type,
            data_dir=args.data_dir,
            keep_temp=args.keep_temp,
            watch=args.watch,
            poll_sec=args.poll_sec,
            idle_sec=args.idle_sec,
            lock_dir=args.lock_dir,
            lock_stale_sec=args.lock_stale_sec,
            ipaethr=args.ipaethr,
            rmsdthr=args.rmsdthr,
            passed_dir=args.passed_dir,
            ca_only=args.ca_only,
            ref_chain=args.ref_chain,
            passed_csv_name=args.passed_csv_name,
            target_chains_csv=args.target_chains,
        )
    elif args.cmd == "fixbb":
        run_fixbb(
            pdb_path=args.pdb,
            chain=args.chain,
            out_prefix=args.out_prefix,
            cyclic=args.cyclic,
            offset_type=args.offset_type,
            design_steps=tuple(args.stage_iters),
            data_dir=args.data_dir,
        )
    elif args.cmd == "hallucination":
        run_hallucination(
            length=args.length,
            rm_aa=args.rm_aa,
            out_prefix=args.out_prefix,
            cyclic=args.cyclic,
            offset_type=args.offset_type,
            pre_soft_iters=args.pre_soft_iters,
            stage_iters=tuple(args.stage_iters),
            data_dir=args.data_dir,
        )

if __name__ == "__main__":
    main()
