# TECH 405: Smart Queue and Time Management Assignments

This repository contains the coursework notebooks and supporting outputs for the call-centre queue project. Each week’s deliverables and run instructions are kept with that week’s assignment.

## Week 7 project-wide technical report

The [project-wide technical report](Project_Week7_Technical_Report.docx) documents
the progression across Weeks 2, 4, 5, and 6 while keeping their distinct targets,
datasets, and experiments separate. Regenerate it after executing the weekly
workflows with `python build_project_week7_report.py`.

The [Week 7 technical evidence pack](report_assets/technical_evidence.md) includes
the implementation-specific pipeline flowchart, copied experiment figures,
verified intermediate results, suggested code excerpts, and a checklist of
screenshots to capture from the executed project.

## Assignments

| Week | Assignment | Main deliverable |
|---|---|---|
| 2 | Exploratory data analysis | [Executed EDA notebook](week%202/Week2_EDA_CallCentre.ipynb) |
| 4 | Feed-forward neural-network classification | [Executed notebook](week%204/week4_neural_network.ipynb) |
| 5 | Sequential service-standard classification with an LSTM | [Executed notebook](week%205/week5_rnn_lstm.ipynb) |
| 6 | Customer-support intent classification | [Executed notebook and Week 6 report details](week%206/README.md) |

## Repository structure

```text
.
├── Project_Week7_Technical_Report.docx
├── build_project_week7_report.py
├── data/
│   └── simulated_call_centre.csv
├── requirements.txt
├── report_assets/
│   ├── technical_evidence.md
│   ├── actual_project_pipeline.png
│   ├── actual_project_pipeline.svg
│   └── existing Week 2/4/5/6 result figures
├── week 2/
│   ├── README.md
│   ├── Week2_EDA_CallCentre.ipynb
│   └── images/
├── week 4/
│   ├── README.md
│   ├── week4_neural_network.ipynb
│   ├── results.json
│   ├── images/
│   └── scripts/
├── week 5/
│   ├── README.md
│   ├── week5_rnn_lstm.ipynb
│   ├── images/
│   └── outputs/
└── week 6/
    ├── README.md
    ├── week6_customer_support_intent_classification.ipynb
    ├── Week6_Customer_Support_Intent_Classification_Report.docx
    ├── build_week7_report.py
    ├── images/
    ├── outputs/
    └── build_report.py
```

Saved chart files are grouped in each assignment’s `images/` folder; the executed notebooks also retain their inline outputs. Week 5 and Week 6 keep machine-readable metrics in their `outputs/` folders. The project-wide report summarizes the separate stage results, including Week 6's duplicate-aware holdout evaluation.

## Setup

Weeks 2, 4, and 5 use the shared dependencies listed in [`requirements.txt`](requirements.txt). From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Week 6 has its own [`requirements.txt`](week%206/requirements.txt) and dataset setup instructions in its [assignment README](week%206/README.md).

## Shared dataset

Weeks 2, 4, and 5 use [`data/simulated_call_centre.csv`](data/simulated_call_centre.csv), a simulated call-centre dataset. Check the dataset’s source and redistribution terms before publishing the CSV in a public repository. Week 6 downloads its Bitext customer-support dataset as described in its assignment README; the raw Week 6 source data is not stored here.
