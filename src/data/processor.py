from pathlib import Path
from PIL import Image
import pydicom
import uuid
import pandas as pd
from typing import Optional, List, Dict

from src.data.metadata import source_columns


def _is_image_file(p: Path):
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".dcm"}


def _read_image_to_png(src: Path, dest: Path, size=None, max_long_edge=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if src.suffix.lower() == ".dcm":
            ds = pydicom.dcmread(str(src))
            arr = ds.pixel_array
            img = Image.fromarray(arr)
            source_format = "DICOM"
        else:
            img = Image.open(src)
            source_format = img.format or src.suffix.lstrip(".").upper()
        source_width, source_height = img.size
        source_mode = img.mode
        img = img.convert('L')
        if size is not None:
            img = img.resize(size)
        elif max_long_edge and max(img.size) > max_long_edge:
            scale = max_long_edge / max(img.size)
            resized_size = (
                max(1, round(img.width * scale)),
                max(1, round(img.height * scale)),
            )
            img = img.resize(resized_size, resample=Image.Resampling.LANCZOS)
        img.save(dest)
        return {
            "source_image_width": str(source_width),
            "source_image_height": str(source_height),
            "source_image_mode": source_mode,
            "source_image_format": source_format,
            "image_width": str(img.width),
            "image_height": str(img.height),
            "image_mode": "L",
            "image_format": "PNG",
        }
    except Exception:
        return None


def _read_report_text(txt_path: Path):
    try:
        return txt_path.read_text(encoding='utf-8').strip()
    except Exception:
        return ""


def _source_path(src_path: Path, raw_dir: Path) -> str:
    try:
        return str(src_path.resolve().relative_to(raw_dir.resolve()))
    except ValueError:
        return str(src_path.resolve())


def _row_value(row, names) -> str:
    names = {name.lower() for name in names}
    for column, value in row.items():
        if column.lower() in names and pd.notna(value):
            return str(value).strip()
    return ""


def _report_fields(row) -> Dict[str, str]:
    findings = _row_value(row, ["findings"])
    impression = _row_value(row, ["impression"])
    report = _row_value(row, ["report_text", "report", "final", "abstract", "comments"])
    report_parts = []
    for value in [findings, impression, report]:
        if value and value not in report_parts:
            report_parts.append(value)
    return {
        "findings": findings,
        "impression": impression,
        "report_text": "\n\n".join(report_parts),
    }


def process_iuxray_directory(raw_dir: Path, out_dir: Path, image_size=None,
                            max_long_edge: Optional[int] = 1024,
                            projections_csv: Optional[Path] = None,
                            reports_csv: Optional[Path] = None) -> List[Dict]:
    """Process IU Xray data into a processed folder of PNGs and metadata.

    Behavior:
    - If `projections_csv` and `reports_csv` are provided (or found under `raw_dir`),
      use them to select only frontal view images and attach report metadata to each image.
    - Otherwise fall back to a heuristic folder-based pairing using .txt files and image basenames.

    By default, images preserve their original aspect ratio. Set ``image_size``
    only for an explicit fixed-size export.

    Returns a list of metadata dicts: {id, image_path, report_text, **other_metadata}
    """
    # Try to locate CSVs if not explicitly provided
    if projections_csv is None:
        p = raw_dir / 'indiana_projections.csv'
        if p.exists():
            projections_csv = p
    if reports_csv is None:
        r = raw_dir / 'indiana_reports.csv'
        if r.exists():
            reports_csv = r

    if projections_csv and reports_csv and projections_csv.exists() and reports_csv.exists():
        return _process_from_csvs(
            raw_dir,
            out_dir,
            projections_csv,
            reports_csv,
            image_size,
            max_long_edge,
        )

    # Fallback: original heuristic
    metadata = []
    image_out_dir = out_dir / "images"
    image_out_dir.mkdir(parents=True, exist_ok=True)

    for folder in raw_dir.rglob('*'):
        if folder.is_dir():
            imgs = [p for p in folder.iterdir() if p.is_file() and _is_image_file(p)]
            txts = [p for p in folder.iterdir() if p.suffix.lower() == '.txt']

            if not imgs and not txts:
                continue

            report_text = ""
            if txts:
                texts = [ _read_report_text(t) for t in txts ]
                report_text = "\n\n".join([t for t in texts if t])

            for img in imgs:
                unique_id = str(uuid.uuid4())
                out_img_name = f"{unique_id}.png"
                out_img_path = image_out_dir / out_img_name
                image_info = _read_image_to_png(
                    img,
                    out_img_path,
                    size=image_size,
                    max_long_edge=max_long_edge,
                )
                if image_info is None:
                    continue

                if not report_text:
                    same_txt = img.with_suffix('.txt')
                    if same_txt.exists():
                        report_text = _read_report_text(same_txt)

                metadata.append({
                    'id': unique_id,
                    'image_path': str(out_img_path.relative_to(out_dir)),
                    'dataset_name': 'IU-Xray',
                    'ingestion_mode': 'folder_heuristic',
                    'source_image_path': _source_path(img, raw_dir),
                    'source_filename': img.name,
                    'report_text': report_text,
                    'report_char_count': str(len(report_text)),
                    'report_word_count': str(len(report_text.split())),
                    'has_report': str(bool(report_text)).lower(),
                    **image_info,
                })

    return metadata


def _process_from_csvs(raw_dir: Path, out_dir: Path, projections_csv: Path,
                       reports_csv: Path, image_size=None,
                       max_long_edge: Optional[int] = 1024) -> List[Dict]:
    """Process IU Xray using the indiana projections + reports CSVs.

    - Filters to only frontal/projection views (PA/AP/frontal).
    - Expects projections CSV to contain a column with filename/path and a column describing the view.
    - Expects reports CSV to contain a `uid` column and text fields like `impression` or `findings`.
    """
    df_proj = pd.read_csv(projections_csv)
    df_rep = pd.read_csv(reports_csv)

    # normalize column names
    proj_cols = {c.lower(): c for c in df_proj.columns}
    rep_cols = {c.lower(): c for c in df_rep.columns}

    # find likely filename/path column in projections
    filename_col = None
    for candidate in ['filename','file','path','image','imagepath','image_name']:
        if candidate in proj_cols:
            filename_col = proj_cols[candidate]
            break
    # fallback: choose first column containing typical image extensions
    if filename_col is None:
        for c in df_proj.columns:
            sample = str(df_proj[c].astype(str).dropna().iloc[0]) if not df_proj[c].dropna().empty else ''
            if any(ext in sample.lower() for ext in ['.dcm','.png','.jpg','.jpeg']):
                filename_col = c
                break

    # find view/projection column
    view_col = None
    for candidate in ['view','projection','projection_name','projection_type','exam_type','image']:
        if candidate in proj_cols and candidate != filename_col:
            view_col = proj_cols[candidate]
            break
    # fallback: try columns containing strings like 'PA' in samples
    if view_col is None:
        for c in df_proj.columns:
            sample_vals = df_proj[c].dropna().astype(str).head(20).str.lower().tolist()
            if any(('pa' in v or 'ap' in v or 'frontal' in v or 'anteroposterior' in v or 'posteroanterior' in v) for v in sample_vals):
                view_col = c
                break

    # uid column to join with reports
    uid_col_proj = None
    uid_col_rep = None
    for candidate in ['uid','study_id','patient_id','id']:
        if candidate in proj_cols:
            uid_col_proj = proj_cols[candidate]
        if candidate in rep_cols:
            uid_col_rep = rep_cols[candidate]

    # if we have uid on both, merge; else try to match by filename/identifier
    merged = None
    if uid_col_proj and uid_col_rep:
        merged = df_proj.merge(df_rep, left_on=uid_col_proj, right_on=uid_col_rep, how='left', suffixes=('_proj','_rep'))
    else:
        # attempt merge on a column named 'image' in reports that may contain view text; fallback to concatenation
        merged = df_proj.copy()
        # attach report text if reports file has uid present in proj under a column
        # simplest: left-join on any overlapping column names
        common = set(df_proj.columns).intersection(set(df_rep.columns))
        if common:
            common_col = list(common)[0]
            merged = df_proj.merge(df_rep, on=common_col, how='left')
        else:
            # no reliable join; attach empty report fields
            for c in df_rep.columns:
                merged[c] = None

    # Define frontal view acceptance
    def is_frontal(s: str) -> bool:
        if not isinstance(s, str):
            return False
        s_low = s.lower()
        return any(token in s_low for token in ['pa','ap','frontal','anteroposterior','posteroanterior'])

    image_out_dir = out_dir / 'images'
    image_out_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for _, row in merged.iterrows():
        # pick view value
        view_val = None
        if view_col and pd.notna(row.get(view_col)):
            view_val = str(row.get(view_col))
        else:
            # try common report image column
            for c in ['image','exam','study']:
                if c in merged.columns and pd.notna(row.get(c)):
                    view_val = str(row.get(c))
                    break

        if not is_frontal(view_val):
            continue

        # determine source image path
        src_path = None
        if filename_col and pd.notna(row.get(filename_col)):
            candidate = str(row.get(filename_col))
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute():
                # try relative to raw_dir
                candidate_path = raw_dir / candidate
            if candidate_path.exists():
                src_path = candidate_path
            else:
                # sometimes filenames are just ids; try to find matching file in raw_dir
                files = list(raw_dir.rglob(f"{candidate}*"))
                if files:
                    src_path = files[0]

        # if no src found, try to locate any image under a directory matching uid
        if src_path is None:
            # try uid from projections
            for col in merged.columns:
                if col.lower() in ['uid','id','study_id','patient_id'] and pd.notna(row.get(col)):
                    uid = str(row.get(col))
                    files = list(raw_dir.rglob(f"*{uid}*"))
                    img_files = [f for f in files if _is_image_file(f)]
                    if img_files:
                        src_path = img_files[0]
                        break

        if src_path is None or not src_path.exists():
            continue

        # process and save image
        unique_id = str(uuid.uuid4())
        out_img_name = f"{unique_id}.png"
        out_img_path = image_out_dir / out_img_name
        image_info = _read_image_to_png(
            src_path,
            out_img_path,
            size=image_size,
            max_long_edge=max_long_edge,
        )
        if image_info is None:
            continue

        # collect metadata fields from the merged row
        meta = {
            'id': unique_id,
            'image_path': str(out_img_path.relative_to(out_dir)),
            'dataset_name': 'IU-Xray',
            'ingestion_mode': 'csv_join',
            'source_image_path': _source_path(src_path, raw_dir),
            'source_filename': src_path.name,
            'projection': view_val or '',
            'uid': _row_value(row, ['uid', 'study_id', 'patient_id', 'id']),
            **image_info,
            **_report_fields(row),
            **source_columns(row),
        }
        meta['report_char_count'] = str(len(meta['report_text']))
        meta['report_word_count'] = str(len(meta['report_text'].split()))
        meta['has_report'] = str(bool(meta['report_text'])).lower()

        metadata.append(meta)

    return metadata
