# UAV Low-Light Detection

## 1. Install

```bash
conda create -n uavllie python=3.10 -y
conda activate uavllie
pip install -r requirements.txt
```

## 2. Convert VisDrone to YOLO

Unzip VisDrone into this raw structure:

```text
raw/VisDrone/
  VisDrone2019-DET-train/
    images/
    annotations/
  VisDrone2019-DET-val/
    images/
    annotations/
  VisDrone2019-DET-test-dev/
    images/
    annotations/
```

Run:

```bash
python src/preprocess/visdrone2yolo.py \
  --raw-root raw/VisDrone \
  --out-root datasets/VisDrone
```

Check:

```bash
python src/preprocess/check_yolo_dataset.py --data datasets/VisDrone/visdrone_clean.yaml
```

Draw one label:

```bash
python src/preprocess/draw_yolo_labels.py \
  --image datasets/VisDrone/images/val/YOUR_IMAGE.jpg \
  --out debug_label.jpg
```

## 3. Test one image low-light degradation

```bash
python src/preprocess/test_one_image_lowlight.py \
  --image datasets/VisDrone/images/val/YOUR_IMAGE.jpg \
  --level LL2 \
  --outdir debug_one_image
```

Output:

```text
debug_one_image/YOUR_IMAGE_LL2.jpg
debug_one_image/YOUR_IMAGE_LL2_gamma.jpg
debug_one_image/YOUR_IMAGE_LL2_clahe.jpg
debug_one_image/YOUR_IMAGE_comparison.jpg
```

## 4. Generate full LowLight-VisDrone

```bash
python src/preprocess/make_lowlight_visdrone.py \
  --input-root datasets/VisDrone \
  --output-root datasets/VisDrone-LL/LL1 \
  --level LL1 \
  --seed 202606031

python src/preprocess/make_lowlight_visdrone.py \
  --input-root datasets/VisDrone \
  --output-root datasets/VisDrone-LL/LL2 \
  --level LL2 \
  --seed 202606032

python src/preprocess/make_lowlight_visdrone.py \
  --input-root datasets/VisDrone \
  --output-root datasets/VisDrone-LL/LL3 \
  --level LL3 \
  --seed 202606033
```

## 5. Apply gamma / CLAHE baseline to one split

```bash
python src/preprocess/enhancement_baselines.py \
  --input datasets/VisDrone-LL/LL2/images/val \
  --output datasets/VisDrone-LL/LL2_gamma/images/val \
  --method gamma \
  --gamma 0.60

python src/preprocess/enhancement_baselines.py \
  --input datasets/VisDrone-LL/LL2/images/val \
  --output datasets/VisDrone-LL/LL2_clahe/images/val \
  --method clahe \
  --clahe-clip 2.0
```

Remember to copy labels unchanged if you create a full dataset variant:

```bash
mkdir -p datasets/VisDrone-LL/LL2_gamma/labels
cp -r datasets/VisDrone-LL/LL2/labels/* datasets/VisDrone-LL/LL2_gamma/labels/
```

## 6. Recommended dataset roles

- Main detection: VisDrone -> LowLight-VisDrone.
- Enhancement pretrain: LOL.
- Cross-domain low-light detection: ExDark.
- Strong UAV low-light cross test: DroneVehicle RGB subset, if downloadable.
