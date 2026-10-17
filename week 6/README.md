# Week 6: Customer Support Intent Classification

This assignment extends the Smart Queue project with an intent-routing task: classify an incoming customer-support request into one of 27 intents.

The executed notebook compares a TF-IDF feed-forward neural network (MLP) and a PyTorch LSTM using the same stratified train, validation, and held-out test split. The notebook reports a majority-class baseline and evaluates the models with accuracy, macro-averaged precision/recall/F1, and multiclass ROC-AUC.

## Deliverables

- [`week6_customer_support_intent_classification.ipynb`](week6_customer_support_intent_classification.ipynb) — executed analysis, training, evaluation, and figures.
- [`Week6_Customer_Support_Intent_Classification_Report.docx`](Week6_Customer_Support_Intent_Classification_Report.docx) — formatted assignment report.
- [`images/`](images/) — confusion matrices, ROC curves, and training-loss plots used by the report.
- [`outputs/week6_results.json`](outputs/week6_results.json) — saved evaluation metrics used by the report builder.
- [`build_report.py`](build_report.py) — rebuilds the DOCX report from the saved metrics and figures.
- [`requirements.txt`](requirements.txt) — Python dependencies for the notebook and report builder.

## Run the notebook

Use Python 3.12, which is the version used for the saved notebook execution. From this folder, create an environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Open this folder in VS Code, open the notebook, select the `.venv` Python environment as its kernel, and choose **Run All**. The first run downloads the dataset from Hugging Face into a cache under your user profile; the raw dataset is not stored in this repository. Run the notebook with this folder as its working directory so generated plots go to `images/` and metrics go to `outputs/`.

To rebuild the report after running the notebook:

```powershell
python build_report.py
```

The report builder reads `outputs/week6_results.json` and the three plot images in `images/`, then writes the DOCX file in this folder.

## Dataset and limitations

The notebook uses the [Bitext Customer Service Tagged Training Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Its dataset card declares the CDLA-Sharing-1.0 license and describes the examples as hybrid synthetic data with automated labeling. Review the dataset card and license terms before redistributing the data. Model performance on this dataset does not establish performance on real customer requests.

## Folder structure

```text
week 6/
├── README.md
├── Week6_Customer_Support_Intent_Classification_Report.docx
├── build_report.py
├── images/
│   ├── confusion_matrices.png
│   ├── roc_curves.png
│   └── training_loss.png
├── outputs/
│   └── week6_results.json
├── requirements.txt
├── week6_customer_support_intent_classification.ipynb
```
