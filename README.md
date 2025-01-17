# NNR Classifier
This is my implementation of a Nearest Neighbor Radius (NNR) classification approach in Python. 

## Features
* **Categorical Encoding:** With LabelEncoder.
* **Scaling:** With StandardScaler for numeric data.
* **Distance Calculation:** Vectorized Euclidean distances (efficient with NumPy).
* **Adaptive Radius:** Finds the best radius via validation data, and adjusts it if not enough neighbors are found.

## Quick Start
**1. Clone and Install:**

```bash
git clone https://github.com/nivkapliser/nnr-classifier.git

cd nnr-classifier

pip install -r requirements.txt
```

**2. Data Setup:**

* Training & Validation CSVs must include a `"class"` column.
* Test CSV may omit `"class"` unless you want to evaluate accuracy.

**3. Config File (`config.json` example):**

```json

{
  "data_file_train": "data/train.csv",
  "data_file_validation": "data/validation.csv",
  "data_file_test": "data/test.csv"
}
```

**4. Run:**

```bash
python nnr_classifier.py
```
* Reads config.json to load datasets.
* Performs preprocessing, selects optimal radius, and classifies test data.
* Prints final accuracy (if test labels are present).
  
## Structure
```arduino
nnr-classifier/
├── nnr_classifier.py
├── config.json
├── data/
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
└── README.md
```
