# pneumonia-detection-app
Chest X-ray Pneumonia detection and screening app built with Python, PyTorch/YOLO, and Streamlit
# Pneumonia Screen

AI-assisted chest X-ray screening for pneumonia. Upload a frontal chest X-ray and get one of three outcomes: **No pneumonia detected**, **Uncertain**, or **Pneumonia suspected**. Suspected cases come with boxes around the suspicious regions, a plain-language location, and a downloadable PDF report.

> **Important:** This is a research and learning prototype. It is a screening aid, not a diagnostic device, and it has not been clinically validated. Do not use it to diagnose, start or stop treatment. Every result must be reviewed by a qualified doctor.

![Screenshot](docs/screenshot.png)

## Features

- Upload PNG, JPG or DICOM (`.dcm`) chest X-rays
- Automatic check that rejects colour photos that are not X-rays
- Three-outcome result with a 0 to 100 screening score
- Numbered boxes on suspicious regions with a location such as "Right lung, lower zone"
- Adjustable box sensitivity and optional patient details
- One-page PDF report: result, images, regions table, measured model reliability, limitations and disclaimer
- Download of the marked image as PNG

## How it works

Two models work together:

| Model | Job | Details |
|---|---|---|
| DenseNet121 classifier | Is there pneumonia? | ImageNet-pretrained, 512 px, mixed precision, augmentation, flip test-time averaging |
| YOLOv8s detector | Where is it? | 640 px, 40 epochs, draws boxes on pneumonia-like opacities |

Each model's score is converted to a percentile against the validation set, and the two are averaged into one screening score (0 to 100). Two cut-points, chosen on the validation set only, split the score into three zones:

- **No pneumonia:** below the lower cut (at least 95% of validation pneumonia cases scored above it)
- **Uncertain:** between the cuts
- **Pneumonia suspected:** above the upper cut (at least 90% of validation normal cases scored below it)

Boxes are drawn only for "Pneumonia suspected" and "Uncertain" results.

## Data

[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) (1024 x 1024 DICOM chest X-rays with radiologist boxes).

- 6,012 pneumonia images and the same number of randomly chosen non-pneumonia images (12,024 total)
- Split per class: 80% train, 10% validation, 10% test
- The test set (1,202 images) was never used for training or for choosing thresholds
- In this dataset "no pneumonia" includes normal X-rays and X-rays with other abnormalities, so the app says "No pneumonia detected", not "healthy"

The dataset is not included in this repository. Download it from Kaggle and follow the competition rules.

## Results

Measured on 1,202 test X-rays the models never saw (601 pneumonia, 601 non-pneumonia):

| Model | Sensitivity | Specificity | Accuracy | AUC |
|---|---|---|---|---|
| DenseNet121 classifier | 82.0% | 74.7% | 78.4% | 0.872 |
| YOLOv8s detector (image level) | 84.7% | 73.2% | 79.0% | 0.865 |
| **Combined** | **85.7%** | **73.5%** | **79.6%** | **0.878** |

Detector box quality (validation): precision 0.43, recall 0.53, mAP@0.5 0.44.

With the three decision zones on the test set:

| Outcome | Images | What was true |
|---|---|---|
| No pneumonia | 389 | 8.7% still had pneumonia |
| Pneumonia suspected | 465 | 83.9% truly had pneumonia |
| Uncertain | 348 (29%) | about half had pneumonia |

When the system gave a confident answer (71% of images), it was correct 87.2% of the time.

## Project structure

```
pneumonia_app/
  app.py              Streamlit user interface
  engine.py           Loads the models and runs them
  utils.py            Decision logic, DICOM loading, box drawing
  report.py           PDF report (ReportLab)
  requirements.txt
  weights/
    classifier.pt     Trained classifier
    cls_config.json   Classifier settings and test metrics
    best.pt           Trained YOLOv8 detector
    config.json       Detector settings and test metrics
    combo_config.json Combined-score thresholds and validation scores
  docs/
    screenshot.png
```

## Installation and usage

Requires Python 3.10 to 3.12.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Place the five weight files in the `weights/` folder first. The app opens in your browser. Upload an X-ray, review the result, and download the PDF report.

A GPU is not required. The app runs on CPU, a little slower.

## Reproducing the training

Training was done in a Kaggle notebook on an NVIDIA T4 GPU:

1. Download the RSNA data with the Kaggle API and convert DICOM files to 640 px PNGs with YOLO-format labels.
2. Train the DenseNet121 classifier (12 epochs, best epoch chosen by validation AUC).
3. Train the YOLOv8s detector (40 epochs).
4. Combine both scores on the validation set, choose the thresholds, and measure once on the test set.
5. Save the five files listed above into `weights/`.

## Limitations

- Trained and tested on one dataset. Not validated on other hospitals, scanners, or age groups.
- RSNA labels are noisy, which limits the achievable accuracy.
- It looks only for pneumonia. It does not detect tuberculosis, nodules, tumours or heart problems, and "No pneumonia detected" does not rule out disease.
- Box confidence values are relative rankings, not probabilities of pneumonia.
- Designed for frontal chest X-rays.

## Roadmap

- Validate on independent datasets (for example VinDr-CXR or CheXpert)
- Model ensembles and more training data
- Visual explanations such as Grad-CAM
- Probability calibration
- Secured deployment with login and audit logs

## Acknowledgements

- RSNA and Kaggle for the Pneumonia Detection Challenge dataset
- PyTorch / torchvision for DenseNet121
- Ultralytics for YOLOv8
- Streamlit and ReportLab

## Author

[Your Name] ([LinkedIn or GitHub link])

## License

[Choose a license, for example MIT. Note that the RSNA dataset has its own terms.]
