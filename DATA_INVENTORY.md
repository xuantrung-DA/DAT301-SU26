# Raw data inventory

Generated: 2026-07-16  
Project: `D:\LOW_LIGHT\DAT301-SU26`

This inventory records the download/extraction checkpoint before any new
dataset conversion, low-light synthesis, training, or evaluation. The source
list follows the supplied four-week PDF; the processing authority for the next
phase is `DAT_Proposal.xlsx`.

## Completion status

| Dataset | Raw extraction path | Verified primary content | Status |
|---|---|---:|---|
| VisDrone2019-DET | `datasets/raw/VisDrone` | train 6,471; val 548; test-dev 1,610 images, with the same number of original annotation files | Complete |
| LOL-v1 | `datasets/raw/LOL` | 485 train pairs + 15 evaluation pairs | Complete |
| AU-AIR | `datasets/raw/AU-AIR` | 32,823 images + 1 official `annotations.json` | Complete |
| ExDark | `datasets/raw/ExDark` | 7,363 images + 7,363 annotation files across 12 classes | Complete |
| UAVDT (Kaggle copy listed in the PDF) | `datasets/raw/UAVDT` | 233,460 extracted files; 24,143 train images + 53,676 test images, each with `ann` and `meta` | Complete |

## Validation details

### VisDrone2019-DET

- Original archives are hard-linked into `datasets/raw/archives`; this keeps
  them inside the project inventory without consuming duplicate disk space.
- Original extracted layout:
  - train: 6,471 images and 6,471 annotations;
  - val: 548 images and 548 annotations;
  - test-dev: 1,610 images and 1,610 annotations.
- Existing YOLO copy at `datasets/VisDrone` was audited independently:
  - train: 343,197 boxes, 0 errors;
  - val: 38,759 boxes, 0 errors;
  - test: 75,101 boxes, 0 errors;
  - image-stem leakage across all split pairs: 0.
- `datasets/VisDrone-LL` is an older generated copy. It is not accepted as the
  final proposal-compliant low-light dataset and must be regenerated in the
  processing phase.

### LOL-v1

- `our485/low` and `our485/high`: 485 image pairs, exact basename match.
- `eval15/low` and `eval15/high`: 15 image pairs, exact basename match.
- `.DS_Store` and `__MACOSX` entries are archive metadata, not dataset images.
- `datasets/raw/LOLdataset.zipz4k35r1k.part` is a stale partial download. The
  complete verified `LOLdataset.zip` is present; the partial file was retained
  because this checkpoint does not perform destructive cleanup.

### AU-AIR

- 32,823 extracted JPEG images.
- 32,823 frame records in `annotations.json`.
- Every annotation record resolves to one image; no missing or unannotated
  images.
- The downloaded official JSON contains 132,031 bounding boxes. The proposal
  table states 132,034, so the next phase must use the actual parsed JSON count
  and record this three-box source discrepancy rather than silently changing
  the source data.
- Official category order in the JSON: Human, Car, Truck, Van, Motorbike,
  Bicycle, Bus, Trailer.

### ExDark

- 7,363 primary images and 7,363 primary annotation text files.
- Class totals match between images and annotations for all 12 classes.
- The only apparent exact-name mismatch is
  `Bicycle/2015_00391.JPG` versus annotation stem
  `Bicycle/2015_00391.jpg`; it is a case-only filename difference.
- `__MACOSX` contains 14,752 metadata files and must be excluded from all
  manifests and conversions.

### UAVDT

- Kaggle archive download size: 12,094,910,763 bytes.
- Server MD5 verified: `7b46e0ed171b234982fca1a597a60c56`.
- Archive index: 233,460 files, expected uncompressed size 13,551,323,501
  bytes.
- Extracted result: exactly 233,460 files and 13,551,323,501 bytes.
- Train: 24,143 `img`, 24,143 `ann`, and 24,143 `meta` files.
- Test: 53,676 `img`, 53,676 `ann`, and 53,676 `meta` files.
- This Kaggle distribution is already organized as `train/{img,ann,meta}` and
  `test/{img,ann,meta}`. Its converter must target this actual layout rather
  than assume the alternate official MOT folder layout.

## Archive checksums

All hashes below are SHA-256 unless otherwise stated.

| Archive | Bytes | SHA-256 |
|---|---:|---|
| `datasets/raw/archives/VisDrone2019-DET-train.zip` | 1,549,875,511 | `86A77EBA93137BFC16E4993860DE9245B0675C0DBA0D3AB98FB458699E256F84` |
| `datasets/raw/archives/VisDrone2019-DET-val.zip` | 81,638,851 | `ABEEA063037E5D20398837DEB11084E652402A34DDF4F207BDF541A6F2A35EF9` |
| `datasets/raw/archives/VisDrone2019-DET-test-dev.zip` | 311,045,829 | `78B0C5078A14EE43D0B803A354E76016D7260D1704CFD1C2DC821858D839E261` |
| `datasets/raw/LOLdataset.zip` | 347,426,968 | `47D85314B7927470CD48C97E5C1D6C896A56D6217C8E508D0736AFF2D91AADCC` |
| `datasets/raw/archives/auair2019data.zip` | 2,318,071,652 | `BD31F8DE50FD54182B3569D5671AD70CC3FCFE7E562313319EF9810F6E91E72B` |
| `datasets/raw/archives/auair2019annotations.zip` | 4,039,375 | `5CEF55C2717B5FCA01D2423CEAD29BA00913C3C12AA4590D04108D71A12B2F46` |
| `datasets/raw/archives/ExDark.zip` | 1,492,131,597 | `2D21A1EDC004603EF36635E08C5E5AA5B4F41B95863B083883F41AD1DD116779` |
| `datasets/raw/archives/ExDark_Anno.zip` | 5,080,194 | `37E535B71D2E43942F6DBAAD1925DD325AFF9873B83B498F823F20C375B7D35D` |
| `datasets/raw/UAVDT/1.archive` | 12,094,910,763 | `9DF54B2C33F2CC30189BDA8741AE9356B18F2EBCF4911947D6E2F40A0C52F761` |

## Next phase (not run at this checkpoint)

Process each dataset with its proposal-specific protocol: VisDrone clean split
and reproducible LL1/LL2/LL3/LLMix synthesis; LOL paired enhancer warm-up only;
AU-AIR sequence-safe cross-domain splitting and overlap mapping; ExDark
real-low-light evaluation with separate full/overlap metrics; and UAVDT as the
auxiliary dataset listed by the supplied PDF.
