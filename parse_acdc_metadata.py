"""
Parse ACDC Info.cfg files to build a per-patient metadata table used by
extract_biomarkers.py and train_diagnosis_classifier.py.

Expected raw ACDC layout (one folder per patient):
    patientXXX/
        Info.cfg
        patientXXX_4d.nii.gz
        patientXXX_frameYY.nii.gz       (YY = ED frame, from Info.cfg)
        patientXXX_frameYY_gt.nii.gz
        patientXXX_frameZZ.nii.gz       (ZZ = ES frame, from Info.cfg)
        patientXXX_frameZZ_gt.nii.gz

Info.cfg is a "key: value" text file with keys typically including:
    ED, ES, Group, Height, Weight, NbFrame

Group is one of: NOR (normal), MINF (prior myocardial infarction),
DCM (dilated cardiomyopathy), HCM (hypertrophic cardiomyopathy),
RV (abnormal right ventricle).

Usage:
    python parse_acdc_metadata.py --raw_root /path/to/ACDC/training \
        --out_csv ./data/acdc_patient_metadata.csv
"""
import argparse
import csv
import re
from pathlib import Path

import nibabel as nib


def parse_info_cfg(path):
    info = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            info[key.strip()] = value.strip()
    return info


def patient_id_from_dirname(name):
    m = re.fullmatch(r"patient(\d{3})", name)
    if not m:
        raise ValueError(f"Cannot parse patient id from directory name: {name}")
    return int(m.group(1))


def get_spacing(nii_path):
    """Returns (spacing_x_mm, spacing_y_mm, spacing_z_mm) from a NIfTI header."""
    img = nib.load(str(nii_path))
    zooms = img.header.get_zooms()
    sx, sy, sz = float(zooms[0]), float(zooms[1]), float(zooms[2])
    return sx, sy, sz


def find_frame_image(patient_dir, frame_idx):
    """ACDC frame files are 2-digit zero padded, but be tolerant of variants."""
    stem = patient_dir.name
    candidates = [
        patient_dir / f"{stem}_frame{frame_idx:02d}.nii.gz",
        patient_dir / f"{stem}_frame{frame_idx:02d}.nii",
        patient_dir / f"{stem}_frame{frame_idx:d}.nii.gz",
        patient_dir / f"{stem}_frame{frame_idx:d}.nii",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def split_for(pid, train_end, val_end):
    if pid <= train_end:
        return "train"
    if pid <= val_end:
        return "val"
    return "test"


def main(args):
    raw_root = Path(args.raw_root)
    patient_dirs = sorted(
        [p for p in raw_root.rglob("patient*") if p.is_dir() and re.fullmatch(r"patient\d{3}", p.name)]
    )
    if not patient_dirs:
        raise RuntimeError(f"No patient*/ directories found under {raw_root}")

    rows = []
    skipped = []
    for pdir in patient_dirs:
        pid = patient_id_from_dirname(pdir.name)
        cfg_path = pdir / "Info.cfg"
        if not cfg_path.exists():
            skipped.append((pid, "missing Info.cfg"))
            continue

        info = parse_info_cfg(cfg_path)
        required = ["ED", "ES", "Group"]
        if any(k not in info for k in required):
            skipped.append((pid, f"Info.cfg missing one of {required}"))
            continue

        ed_frame = int(info["ED"])
        es_frame = int(info["ES"])
        group = info["Group"].strip().upper()

        ed_img_path = find_frame_image(pdir, ed_frame)
        es_img_path = find_frame_image(pdir, es_frame)
        if ed_img_path is None or es_img_path is None:
            skipped.append((pid, f"ED/ES frame image not found ({ed_frame}/{es_frame})"))
            continue

        sx, sy, sz = get_spacing(ed_img_path)

        rows.append({
            "patient_id": pid,
            "patient": pdir.name,
            "group": group,
            "ed_frame": ed_frame,
            "es_frame": es_frame,
            "ed_image_path": str(ed_img_path),
            "es_image_path": str(es_img_path),
            "height_cm": info.get("Height", ""),
            "weight_kg": info.get("Weight", ""),
            "nb_frame": info.get("NbFrame", ""),
            "spacing_x_mm": sx,
            "spacing_y_mm": sy,
            "spacing_z_mm": sz,
            "split": split_for(pid, args.train_end, args.val_end),
        })

    rows.sort(key=lambda r: r["patient_id"])

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Parsed metadata for {len(rows)} patients -> {out_path}")

    groups = {}
    splits = {}
    for r in rows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    print("Class distribution:", groups)
    print("Split distribution:", splits)

    if splits.get("test", 0) == 0:
        print(
            "\nWARNING: 0 patients fell into the 'test' split with train_end="
            f"{args.train_end}, val_end={args.val_end}. This mirrors the split logic "
            "in prepare_acdc.py / train_acdc.py -- if your raw_root only contains the "
            "100 labeled ACDC training patients (ids 1-100), no patient id exceeds "
            "val_end, so the test split is empty. Either lower val_end so it reserves "
            "a held-out slice of the 100 patients, or point raw_root at a directory "
            "that includes additional labeled patients."
        )

    if skipped:
        print(f"\nSkipped {len(skipped)} patients:")
        for pid, reason in skipped:
            print(f"  patient{pid:03d}: {reason}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", required=True, help="Root dir containing patientXXX/ folders")
    p.add_argument("--out_csv", default="./data/acdc_patient_metadata.csv")
    p.add_argument("--train_end", type=int, default=80,
                    help="Same convention as prepare_acdc.py: patients with id <= this go to 'train'")
    p.add_argument("--val_end", type=int, default=100,
                    help="Patients with train_end < id <= val_end go to 'val'; above that, 'test'")
    main(p.parse_args())
