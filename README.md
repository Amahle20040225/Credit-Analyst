# FNB DataQuest 2026: Interpretable Credit Modeling Console

A Streamlit-based interactive credit risk modeling application for analyzing and predicting credit risk using machine learning.

## Prerequisites

- **Python 3.11 (recommended)** installed on your system — or use a Conda environment (preferred on Windows)
- **pip** package manager (use `python -m pip` to ensure the correct interpreter)

## Installation

### 1. Clone or Download the Repository

```bash
cd fnb_dataquests
```

### 2. Create and Activate a Python Virtual Environment (Recommended)

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

> Use `python -m pip` so that dependencies install into the exact Python interpreter you will use to run the app.

Note: `scikit-learn` is no longer required in `requirements.txt`. The application provides a pure-Python fallback logistic engine when `scikit-learn` is not installed. If you want the faster, production-ready `scikit-learn` implementation, install it separately (Conda recommended on Windows).

## Running the Application

### Option A: Simple Run (Windows, Mac, Linux)

```bash
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

The app will be available at `http://localhost:8501`.

### Option B: If Browser Does Not Open Automatically

Open your browser manually and go to:

```bash
http://localhost:8501
```

### Option C: Use a Fixed Port and Network Access

```bash
python -m streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

If you want to bind only to the local machine, change `0.0.0.0` to `127.0.0.1`.

### Option B: Direct Python Execution

```bash
python -m streamlit run app.py
```

### Option C: Windows (if streamlit not in PATH)

If you get a "streamlit not found" error, use the full Python path:

```powershell
# On Windows (replace with your Python path if different)
python -m streamlit run app.py
```

## Features

- **� Interface 0:** Credit Risk Performance
- **📊 Interface 1:** Customer Risk Insight
- **🔄 Interface 2:** Risk Score Analysis
- **🎯 Interface 3:** Risk Prediction & Strategy Optimization

## Project Structure

```
fnb_dataquests/
├── app.py                 # Main Streamlit application
├── data_utils.py          # Data processing & cleaning utilities
├── model_engine.py        # Credit risk model engine
├── loan_book.csv          # Dataset (3 years historical data)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Data

The application uses `loan_book.csv` containing 3 years of historical loan application data with features including:
- Applicant demographics (age, income, employment length)
- Credit history (delinquencies, inquiries, account age)
- Loan characteristics (amount, type, purpose)
- Risk indicators (DTI ratio, credit utilization)

## Troubleshooting

### Port Already in Use

If port 8501 is already in use:

```bash
streamlit run app.py --server.port 8502
```

### Module Import Errors & scikit-learn build failures (Windows)

Symptom: installing `scikit-learn` from source fails with long compiler output (Meson / MSVC / ninja). This commonly happens when the Python version on the target machine does not have a prebuilt wheel available (for example, newer Python 3.14 wheels may be missing) and pip attempts to compile C/C++ extensions.

Recommended fixes (choose one):

- Easiest — use Conda (Windows):

```powershell
conda create -n fnb_py311 python=3.11 -y
conda activate fnb_py311
conda install -c conda-forge scikit-learn=1.3 streamlit pandas numpy matplotlib seaborn -y
python -m pip install -r requirements.txt --no-deps
```

- Or use a Python 3.11 virtualenv and pip (cross-platform):

```powershell
# check python version
python --version

# create venv and activate (Windows PowerShell)
python -m venv .venv
.\.venv\\Scripts\\Activate.ps1

# upgrade pip build tooling and install
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

- If you must keep Python >= 3.12/3.14 and pip tries to compile scikit-learn, either:
	- install the Microsoft C++ Build Tools / Visual Studio components and Ninja (slow and error-prone), or
	- install scikit-learn from conda-forge (recommended) because it provides prebuilt binaries for many Python versions.

Notes:
- Using `python -m pip` ensures packages are installed into the same interpreter used to run Streamlit.
- Prefer Conda on Windows because it provides prebuilt native wheels and avoids complex compiler toolchains.

### CSV File Not Found

Ensure `loan_book.csv` is in the same directory as `app.py`

## Author

**Mivuyo Xinindlu** - NMU Final Year (2026)
**Amahle Mbenguzna** - NMU IT GRADUATE (2026)

## License

MIT License - Please dont use and modify without authority.
