# Week 6: Customer Support Intent Classification

This assignment extends the Smart Queue project with an intent-routing task: classify an incoming customer-support request into one of 27 intents.

The executed notebook compares a TF-IDF feed-forward neural network (MLP) and a PyTorch LSTM using the same duplicate-aware, intent-stratified train, validation, and held-out test split. Case- and whitespace-normalized duplicate requests are kept in a single partition. The notebook reports a majority-class baseline and evaluates the models with accuracy, macro-averaged precision/recall/F1, and multiclass ROC-AUC.

## Deliverables

- [`week6_customer_support_intent_classification.ipynb`](week6_customer_support_intent_classification.ipynb) — executed analysis, training, evaluation, and figures.
- [`Week7_Assignment6_Technical_Report.docx`](Week7_Assignment6_Technical_Report.docx) — editable Week 7 report focused on the actual Assignment 6 intent-classification experiment.
- [`build_assignment6_week7_report.py`](build_assignment6_week7_report.py) — rebuilds the Assignment 6 Week 7 report and its implementation-specific pipeline flowchart from the saved notebook, results, and figures.
- [`Week6_Customer_Support_Intent_Classification_Report.docx`](Week6_Customer_Support_Intent_Classification_Report.docx) — Week 6-specific report with the duplicate-aware experiment, EDA, code excerpts, results, and figures.
- [`images/`](images/) — EDA, confusion-matrix, ROC, and training-loss figures used by the report.
- [`outputs/week6_results.json`](outputs/week6_results.json) — saved evaluation metrics used by the report builder.
- [`build_week7_report.py`](build_week7_report.py) — rebuilds the Week 6-specific report from its saved metrics, executed notebook, and figures.
- [`requirements.txt`](requirements.txt) — Python dependencies for the notebook and report builder.

## Run the notebook

From this folder, create an environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open this folder in VS Code, open the notebook, select the `.venv` Python environment as its kernel, and choose **Run All**. The first run downloads the dataset from Hugging Face into a cache under your user profile; the raw dataset is not stored in this repository. Run the notebook with this folder as its working directory so generated plots go to `images/` and metrics go to `outputs/`.

To rebuild the Week 6-specific report after running the notebook, run this from the repository root:

```powershell
python "week 6/build_week7_report.py"
```

The report builder reads `week 6/outputs/week6_results.json`, the executed notebook, and the plot images. It updates this folder's report and `Week6_Intent_Classification_Report_APA7_Final.docx` at the repository root. The project-wide Week 7 report covering all project stages is built separately with [`build_project_week7_report.py`](../build_project_week7_report.py).

To rebuild the comprehensive Assignment 6 Week 7 technical report, run:

```powershell
python "week 6/build_assignment6_week7_report.py"
```

This creates `Week7_Assignment6_Technical_Report.docx` in this folder and saves its editable vector/raster pipeline figure in the repository's `report_assets/` folder. The report uses the recorded results and saved experimental figures; it does not fabricate user-interface screenshots.

## Dataset and limitations

The notebook uses the [Bitext Customer Service Tagged Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Its dataset card declares the CDLA-Sharing-1.0 license and describes the examples as hybrid synthetic data with automated labeling. The notebook found repeated request text and groups case/whitespace-normalized duplicates before splitting; paraphrase or template overlap is not fully assessed. Review the dataset card and license terms before redistributing the data. Model performance on this dataset does not establish performance on real customer requests.

## Folder structure

```text
week 6/
├── README.md
├── Week6_Customer_Support_Intent_Classification_Report.docx
├── build_report.py
├── build_week7_report.py
├── images/
│   ├── intent_class_distribution_and_duplicates.png
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── training_loss.png
├── outputs/
│   └── week6_results.json
├── requirements.txt
├── week6_customer_support_intent_classification.ipynb
```
