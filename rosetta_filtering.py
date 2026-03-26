#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

import pyrosetta
from pyrosetta import rosetta


def cyclize_chain_head_to_tail(pose: rosetta.core.pose.Pose, chain_letter: str = "B") -> None:
    """Head-to-tail cyclize chain_letter: connect last residue C to first residue N."""
    chain_sel = rosetta.core.select.residue_selector.ChainSelector(chain_letter)
    subset = chain_sel.apply(pose)
    idxs = [i for i in range(1, pose.total_residue() + 1) if subset[i]]
    if len(idxs) < 2:
        raise ValueError(f"Chain {chain_letter} has <2 residues; cannot cyclize.")

    first_i, last_i = idxs[0], idxs[-1]

    # Remove terminus variants so Rosetta can make a polymer bond
    rosetta.core.pose.remove_lower_terminus_type_from_pose_residue(pose, first_i)
    rosetta.core.pose.remove_upper_terminus_type_from_pose_residue(pose, last_i)

    # Declare C(last) - N(first) chemical bond
    pose.conformation().declare_chemical_bond(last_i, "C", first_i, "N")


def make_interface_selectors(chainA: str = "A", chainB: str = "B", dist: float = 14.0):
    """interface = (B within dist of A) OR (A within dist of B)."""
    target = rosetta.core.select.residue_selector.ChainSelector(chainA)
    binder = rosetta.core.select.residue_selector.ChainSelector(chainB)

    near_target = rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    near_target.set_focus_selector(target)
    near_target.set_distance(dist)
    near_target.set_include_focus_in_subset(False)

    near_binder = rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    near_binder.set_focus_selector(binder)
    near_binder.set_distance(dist)
    near_binder.set_include_focus_in_subset(False)

    binder_iface = rosetta.core.select.residue_selector.AndResidueSelector(binder, near_target)
    target_iface = rosetta.core.select.residue_selector.AndResidueSelector(target, near_binder)
    interface = rosetta.core.select.residue_selector.OrResidueSelector(binder_iface, target_iface)

    return interface, binder, target


def minimize_interface_cartesian(
    pose: rosetta.core.pose.Pose,
    interface_selector,
    scorefxn_cart,
    tol: float = 0.01,
    allow_jump: bool = True,
) -> None:
    """Cartesian minimize chi for interface residues; optionally allow RB jumps."""
    subset = interface_selector.apply(pose)

    movemap = rosetta.core.kinematics.MoveMap()
    movemap.set_bb(False)
    movemap.set_chi(False)
    if allow_jump:
        for j in range(1, pose.num_jump() + 1):
            movemap.set_jump(j, True)

    for i in range(1, pose.total_residue() + 1):
        if subset[i]:
            movemap.set_chi(i, True)

    minm = rosetta.protocols.minimization_packing.MinMover()
    minm.movemap(movemap)
    minm.score_function(scorefxn_cart)
    minm.cartesian(True)
    minm.min_type("lbfgs_armijo_nonmonotone")
    minm.tolerance(tol)
    minm.apply(pose)


def compute_ddg_interface_analyzer(
    pose: rosetta.core.pose.Pose, scorefxn, interface_str: str = "A_B"
) -> float:
    """
    Uses InterfaceAnalyzerMover; get_interface_dG() corresponds to Rosetta's interface ΔG estimate.
    """
    iam = rosetta.protocols.analysis.InterfaceAnalyzerMover(interface_str, False, scorefxn)
    iam.set_pack_input(True)
    iam.set_pack_separated(True)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.apply(pose)
    return float(iam.get_interface_dG())


def compute_sap_metric(pose: rosetta.core.pose.Pose, binder_selector) -> float:
    """
    SapScoreMetric location can vary slightly by build; this works for many full PyRosetta builds.
    """
    sap = rosetta.core.simple_metrics.per_residue_metrics.SapScoreMetric()
    sap.set_score_selector(binder_selector)
    return float(sap.calculate(pose))


def compute_cms(pose: rosetta.core.pose.Pose, target_selector, binder_selector, distance_weight: float = 0.5) -> float:
    cms = rosetta.protocols.simple_filters.ContactMolecularSurfaceFilter()
    cms.distance_weight(distance_weight)
    cms.target_selector(target_selector)
    cms.binder_selector(binder_selector)
    return float(cms.report_sm(pose))


def analyze_one_pdb(
    pdb_path: Path,
    chainA: str,
    chainB: str,
    iface_dist: float,
    sfxn,
    sfxn_cart,
    allow_jump: bool,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pdb": str(pdb_path),
        "ddg_rosetta": None,
        "sap_macrocycle": None,
        "cms": None,
        "status": "ok",
        "error": "",
    }

    try:
        pose = rosetta.core.import_pose.pose_from_file(str(pdb_path))

        # 1) Cyclize macrocycle chain
        cyclize_chain_head_to_tail(pose, chainB)

        # 2) Interface selector
        interface_sel, binder_sel, target_sel = make_interface_selectors(chainA, chainB, iface_dist)

        # 3) Minimize interface
        minimize_interface_cartesian(pose, interface_sel, sfxn_cart, tol=0.01, allow_jump=allow_jump)

        # 4) Metrics
        out["ddg_rosetta"] = compute_ddg_interface_analyzer(pose, sfxn, interface_str=f"{chainA}_{chainB}")

        # SAP / CMS (may fail if build lacks bindings; keep ddG anyway)
        try:
            out["sap_macrocycle"] = compute_sap_metric(pose, binder_sel)
        except Exception as e:
            out["sap_macrocycle"] = None
            out["status"] = "partial"
            out["error"] += f"SAP_failed: {e}; "

        try:
            out["cms"] = compute_cms(pose, target_sel, binder_sel, distance_weight=0.5)
        except Exception as e:
            out["cms"] = None
            out["status"] = "partial"
            out["error"] += f"CMS_failed: {e}; "

    except Exception as e:
        out["status"] = "failed"
        out["error"] = str(e)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, type=Path, help="Folder with peptide complex PDBs")
    ap.add_argument("--out_prefix", required=True, type=Path, help="Output prefix (no extension)")
    ap.add_argument("--chainA", default="A")
    ap.add_argument("--chainB", default="B")
    ap.add_argument("--iface_dist", type=float, default=14.0)
    ap.add_argument("--allow_jump", action="store_true", help="Allow rigid-body minimization between chains")
    ap.add_argument("--extra_flags", default="", help='Extra PyRosetta init flags, e.g. "-extra_res_fa X.params"')
    args = ap.parse_args()

    # Init once for the whole batch
    flags = f"-mute all {args.extra_flags}".strip()
    pyrosetta.init(flags)

    # Scorefunctions
    sfxn = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16")
    sfxn_cart = rosetta.core.scoring.ScoreFunctionFactory.create_score_function("beta_nov16_cart")

    # Mirror your constraint reweights (safe even if no constraints are present)
    for st in ("coordinate_constraint", "atom_pair_constraint", "dihedral_constraint", "angle_constraint"):
        sfxn_cart.set_weight(getattr(rosetta.core.scoring, st), 1.0)

    pdbs = sorted(args.in_dir.glob("*.pdb"))
    if not pdbs:
        raise SystemExit(f"No .pdb files found in {args.in_dir}")

    rows: List[Dict[str, Any]] = []
    for pdb in pdbs:
        rows.append(
            analyze_one_pdb(
                pdb,
                chainA=args.chainA,
                chainB=args.chainB,
                iface_dist=args.iface_dist,
                sfxn=sfxn,
                sfxn_cart=sfxn_cart,
                allow_jump=args.allow_jump,
            )
        )

    df = pd.DataFrame(rows)

    # Write CSV + XLSX
    csv_path = args.out_prefix.with_suffix(".csv")
    xlsx_path = args.out_prefix.with_suffix(".xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    # Also write a small summary JSON
    summary = {
        "n_total": int(len(df)),
        "n_ok": int((df["status"] == "ok").sum()),
        "n_partial": int((df["status"] == "partial").sum()),
        "n_failed": int((df["status"] == "failed").sum()),
    }
    args.out_prefix.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
