# X-Ray Single Service

## Overview

This service provides automated chest X-ray analysis using multiple deep learning models based on CheXagent’s image encoder (ViT architecture with SigLip). The pipeline is designed to optimize inference by selectively executing models depending on intermediate predictions. It contains 3 models- Normal vs Abnormal model, TB detection model and 18 disease classification model.

---

# Workflow

## Step 1: Metadata Extraction

The system first extracts:

- Gender
- View Position (AP/PA)

from the input chest X-ray image. This metadata is required to call the Normal vs Abnormal model

---

## Step 2: Conditional Execution Pipeline

### Case 1: Gender & View Information Available

1. The **Normal/Abnormal Classification Model** is executed first.
2. If the prediction is **Normal**:
   - TB model is skipped
   - 18-disease classification model is skipped
   - Response is returned directly as **Normal**
3. If the prediction is **Abnormal**:
   - TB model is executed
   - 18-disease classification model is executed
   - Combined results are returned

---

### Case 2: Gender & View Information Not Available

1. TB model is executed directly
2. 18-disease classification model is executed directly

#### Final Decision

- If:
  - TB prediction = Negative
  - No diseases detected in 18-disease classifier

→ Output is returned as **Normal**

Otherwise:

→ Output is returned as **Abnormal** along with disease predictions.