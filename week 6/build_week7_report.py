"""Build the Week 7 technical report from the executed Week 6 experiment."""

import json
import shutil
from datetime import date
from pathlib import Path

import nbformat
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
RESULTS_PATH = ROOT / "outputs" / "week6_results.json"
NOTEBOOK_PATH = ROOT / "week6_customer_support_intent_classification.ipynb"
IMAGE_DIR = ROOT / "images"
REPORT_PATH = REPO_ROOT / "Week6_Intent_Classification_Report_APA7_Final.docx"
CANONICAL_REPORT_PATH = ROOT / "Week6_Customer_Support_Intent_Classification_Report.docx"
GITHUB_URL = "https://github.com/BishwadeepRai/Smart-queue-Time-Management-system"


def add_hyperlink(paragraph, text, url):
    relationship_id = paragraph.part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relationship_id)
    run = OxmlElement("w:r")
    properties = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "0563C1")
    properties.append(color)
    run.append(properties)
    text_element = OxmlElement("w:t")
    text_element.text = text
    run.append(text_element)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def add_caption(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.line_spacing = 1
    paragraph.paragraph_format.space_after = Pt(12)
    run = paragraph.add_run(text)
    run.italic = True
    run.font.size = Pt(9)


def add_code(document, code):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.left_indent = Inches(0.25)
    paragraph.paragraph_format.right_indent = Inches(0.25)
    paragraph.paragraph_format.line_spacing = 1
    paragraph.paragraph_format.space_after = Pt(10)
    run = paragraph.add_run(code)
    run.font.name = "Consolas"
    run.font.size = Pt(8)


def add_figure(document, path, caption):
    if not path.is_file():
        raise FileNotFoundError(f"Required report figure is missing: {path}")
    picture = document.add_paragraph()
    picture.alignment = WD_ALIGN_PARAGRAPH.CENTER
    picture.add_run().add_picture(str(path), width=Inches(6.2))
    add_caption(document, caption)


def add_table(document, headers, rows):
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, value in zip(table.rows[0].cells, headers):
        cell.text = str(value)
        for run in cell.paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(9)
    for row in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, row):
            cell.text = str(value)
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.line_spacing = 1
                for run in paragraph.runs:
                    run.font.size = Pt(9)
    document.add_paragraph()
    return table


def notebook_excerpt(notebook, cell_index, start, end):
    source = "".join(notebook.cells[cell_index].source)
    start_index = source.index(start)
    end_index = source.index(end, start_index)
    return source[start_index:end_index].rstrip()


def fmt_score(value):
    return "Not calculated" if value is None else f"{value:.4f}"


def main():
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"{RESULTS_PATH} is missing. Execute the updated notebook before building the report."
        )
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    metrics = results["metrics"]
    mlp_history = results["mlp_validation_macro_f1_by_epoch"]
    lstm_history = results["lstm_validation_macro_f1_by_epoch"]
    mlp_best_epoch = max(range(len(mlp_history)), key=mlp_history.__getitem__) + 1
    lstm_best_epoch = max(range(len(lstm_history)), key=lstm_history.__getitem__) + 1

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    styles = document.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(12)
    styles["Normal"].paragraph_format.line_spacing = 2
    for style_name in ("Heading 1", "Heading 2"):
        styles[style_name].font.name = "Times New Roman"
        styles[style_name].font.color.rgb = RGBColor(0, 0, 0)

    title = document.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_before = Pt(100)
    title.paragraph_format.line_spacing = 2
    title_run = title.add_run(
        "Routing Customer Requests by Intent:\n"
        "A Technical Report on Duplicate-Aware Neural Model Evaluation"
    )
    title_run.bold = True
    title_run.font.name = "Times New Roman"
    title_run.font.size = Pt(16)
    for line in (
        "Bishwadeep Rai",
        "Westcliff University",
        "TECH 405: Artificial Neural Network and Deep Learning",
        "Professor Aashish Dhakal",
        date.today().strftime("%B %d, %Y"),
    ):
        paragraph = document.add_paragraph(line)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_page_break()
    heading = document.add_heading("Abstract", level=1)
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.add_paragraph(
        "This project studies whether an English customer-support instruction can be "
        "assigned to one of 27 service intents for downstream queue routing. The "
        "Bitext Customer Service Tagged Training Dataset contains 26,872 examples. "
        "A TF-IDF feed-forward multilayer perceptron (MLP) and a PyTorch long "
        "short-term memory (LSTM) network are compared with a majority-class "
        "baseline. An initial audit found repeated customer instructions, so "
        "case- and whitespace-normalized duplicates were grouped before a "
        "stratified 70/15/15 train/validation/test split. The preprocessing "
        "vocabulary is learned from training data only, and the test partition is "
        "reserved for final evaluation. The report documents the dataset, "
        "exploratory analysis, implementation pipeline, code excerpts, held-out "
        "metrics, and saved result figures. Results are specific to this "
        "synthetic, automatically labeled dataset and do not establish real-world "
        "customer-service performance."
    )
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.line_spacing = 2
    paragraph.add_run("Keywords: ").bold = True
    paragraph.add_run("intent classification, neural networks, LSTM, TF-IDF, duplicate-aware evaluation")

    document.add_heading("Problem Statement", level=1)
    document.add_paragraph(
        "The task is multiclass text classification: given only a customer's "
        "incoming request (`instruction`), predict its labeled service intent "
        "(`intent`) among 27 classes. The intended project use is to route a "
        "request to an appropriate service queue. This is not a prediction of "
        "waiting time, service duration, or whether a call meets a service-level "
        "target. The experiment compares an order-insensitive TF-IDF MLP with a "
        "sequence-based LSTM while holding the source dataset and evaluation "
        "partitions constant."
    )

    document.add_heading("Dataset and Exploratory Data Analysis", level=1)
    dataset_paragraph = document.add_paragraph()
    dataset_paragraph.add_run("Dataset: ").bold = True
    dataset_paragraph.add_run(
        "Bitext Customer Service Tagged Training Dataset, a hybrid synthetic "
        "customer-service corpus with automated labels (Bitext, n.d.). "
    )
    add_hyperlink(dataset_paragraph, "Hugging Face dataset page", results["dataset_url"])
    document.add_paragraph(
        f"The source contains {results['source_rows']:,} English requests and "
        f"{results['number_of_intents']} intents. Each record also contains "
        "response/category/metadata fields, but only `instruction` is used as "
        "`X` and `intent` as `y`. The response is not available at routing time; "
        "the coarse `category` is a correlated label shortcut. The raw text "
        "source is downloaded from Hugging Face into a user cache and is not "
        "committed to the repository. The dataset metadata declares the "
        "CDLA-Sharing-1.0 license; its terms should be reviewed before "
        "redistribution."
    )
    document.add_paragraph(
        "The five dataset columns were checked for both null and blank values: "
        + "; ".join(
            f"{column}: {results['missing_values_by_column'][column]} null, "
            f"{results['empty_values_by_column'][column]} blank"
            for column in results["missing_values_by_column"]
        )
        + ". No rows required imputation or deletion. Intent counts "
        f"range from {results['class_count_min']:,} to "
        f"{results['class_count_max']:,} examples, so the source labels are "
        "near-balanced rather than dominated by a single class."
    )
    length_summary = results["instruction_word_count"]
    median_lengths = results["median_instruction_words_by_intent"]
    shortest_median_intent = min(median_lengths, key=median_lengths.get)
    longest_median_intent = max(median_lengths, key=median_lengths.get)
    document.add_paragraph(
        f"Request length averaged {length_summary['mean']:.2f} whitespace-separated "
        f"words (median {length_summary['median']:.0f}; interquartile range "
        f"{length_summary['q25']:.0f}–{length_summary['q75']:.0f}; observed range "
        f"{length_summary['min']}–{length_summary['max']}). Median request length "
        f"varied from {median_lengths[shortest_median_intent]:.0f} words for "
        f"`{shortest_median_intent}` to {median_lengths[longest_median_intent]:.0f} "
        f"words for `{longest_median_intent}`. This is descriptive analysis and "
        "does not establish text length as a causal or sufficient predictor."
    )
    document.add_paragraph(
        f"The EDA found {results['unique_instruction_groups']:,} distinct "
        "case- and whitespace-normalized instruction groups. "
        f"{results['repeated_instruction_groups']:,} groups contain repeated "
        f"rows, representing {results['additional_repeated_instruction_rows']:,} "
        "additional copies beyond the first occurrence of each request. "
        "Repeated normalized requests had consistent intent labels. This finding "
        "motivated the group-disjoint partitioning described below."
    )
    add_figure(
        document,
        IMAGE_DIR / "intent_class_distribution_and_duplicates.png",
        "Figure 1. Four-panel EDA: intent counts, missing values, repeated "
        "normalized request groups, and median request length by intent.",
    )

    document.add_heading("Model Development Pipeline", level=1)
    document.add_paragraph(
        "The executed notebook follows these stages: (a) validate the required "
        "columns and non-empty input/target values; (b) normalize instruction "
        "case and whitespace for grouping; (c) stratify unique instruction "
        "groups by intent into development and held-out partitions; (d) split "
        "the development groups into training and validation; (e) fit each "
        "representation using training rows only; (f) select checkpoints using "
        "validation macro F1; and (g) evaluate each selected model once on the "
        "same held-out test rows."
    )
    document.add_paragraph(
        "The split is stratified at the normalized-request-group level and is "
        "implemented as a 70% training / 30% holdout split followed by an even "
        "validation/test split of the holdout. Each normalized request group "
        "belongs to exactly one partition. Since duplicate groups have different "
        "sizes, percentages of original rows are approximate; all 27 classes "
        "are present in each partition."
    )
    add_table(
        document,
        ["Partition", "Rows", "Request groups", "Share of rows"],
        [
            [
                "Training",
                f"{results['train_rows']:,}",
                f"{results['train_groups']:,}",
                f"{results['train_rows'] / results['source_rows']:.2%}",
            ],
            [
                "Validation",
                f"{results['validation_rows']:,}",
                f"{results['validation_groups']:,}",
                f"{results['validation_rows'] / results['source_rows']:.2%}",
            ],
            [
                "Held-out test",
                f"{results['test_rows']:,}",
                f"{results['test_groups']:,}",
                f"{results['test_rows'] / results['source_rows']:.2%}",
            ],
        ],
    )
    document.add_paragraph(
        "The normalized grouping is an exact-text control after case-folding and "
        "collapsing whitespace. It does not group paraphrases or all semantically "
        "similar templates; those forms of near-duplicate overlap were not "
        "measured and remain a limitation."
    )

    document.add_heading("Important Implementation Excerpts", level=1)
    document.add_paragraph(
        "The following excerpts are taken directly from the executed notebook and "
        "show the duplicate-aware split and the training-only feature fit."
    )
    document.add_heading("Normalized request grouping and stratified split", level=2)
    add_code(
        document,
        notebook_excerpt(
            notebook,
            5,
            "normalized_instructions = (",
            "\nduplicate_group_counts =",
        )
        + "\n\n"
        + notebook_excerpt(
            notebook,
            7,
            "all_group_ids = np.arange",
            "\n\nassert set(train_groups)",
        ),
    )
    document.add_heading("Training-only TF-IDF representation", level=2)
    add_code(
        document,
        notebook_excerpt(
            notebook,
            9,
            "vectorizer = TfidfVectorizer(",
            "\n\nclass SparseTextMLP",
        ),
    )
    document.add_heading("Recurrent model architecture", level=2)
    add_code(
        document,
        notebook_excerpt(
            notebook,
            11,
            "class IntentLSTM",
            "\n\ndef ",
        ),
    )

    document.add_heading("Models and Training Procedure", level=1)
    document.add_paragraph(
        "The majority baseline always predicts the most frequent training intent. "
        "The feed-forward MLP uses lowercased TF-IDF unigrams and bigrams, two "
        "ReLU hidden layers (256 and 128 units), dropout (0.35 and 0.25), and a "
        "27-class output. It is optimized with Adam (learning rate .001, weight "
        "decay .0001) and cross-entropy. The vocabulary is fitted only on "
        "training instructions; sparse batches are densified for the network."
    )
    document.add_paragraph(
        "The PyTorch LSTM uses a training-only token vocabulary, sequences "
        "truncated/padded to 48 tokens, a 64-dimensional embedding, a 64-unit "
        "LSTM, 0.30 dropout, and a 27-class output. It is optimized with Adam "
        "(learning rate .001) and cross-entropy. Both neural models select the "
        "checkpoint with the highest validation macro F1; test data is not used "
        "for model or epoch selection."
    )
    add_table(
        document,
        ["Model", "Input representation", "Architecture", "Selection"],
        [
            [
                "Feed-forward MLP",
                "Training-only TF-IDF unigrams/bigrams",
                "256 → 128 ReLU; dropout; 27 outputs",
                f"Validation macro F1; best epoch {mlp_best_epoch}",
            ],
            [
                "PyTorch LSTM",
                "Training-only token IDs; max length 48",
                "64-d embedding; 64-unit LSTM; dropout; 27 outputs",
                f"Validation macro F1; best epoch {lstm_best_epoch}",
            ],
        ],
    )

    document.add_heading("Model Evaluation and Performance Analysis", level=1)
    document.add_paragraph(
        "The table reports held-out accuracy and macro-averaged precision, recall, "
        "F1, and one-vs-rest ROC-AUC. Macro averages give each intent equal "
        "weight. The baseline ROC-AUC is not calculated because it produces "
        "constant class predictions."
    )
    model_order = ["Majority baseline", "Feed-forward MLP", "PyTorch LSTM"]
    add_table(
        document,
        ["Model", "Accuracy", "Macro precision", "Macro recall", "Macro F1", "Macro OVR AUC"],
        [
            [
                name,
                *(fmt_score(metrics[name][key]) for key in (
                    "accuracy",
                    "precision_macro",
                    "recall_macro",
                    "f1_macro",
                    "roc_auc_ovr_macro",
                )),
            ]
            for name in model_order
        ],
    )
    mlp_f1 = metrics["Feed-forward MLP"]["f1_macro"]
    lstm_f1 = metrics["PyTorch LSTM"]["f1_macro"]
    better_model = "Feed-forward MLP" if mlp_f1 >= lstm_f1 else "PyTorch LSTM"
    document.add_paragraph(
        f"On this held-out partition, the {better_model} has the higher macro F1 "
        f"({max(mlp_f1, lstm_f1):.4f} versus {min(mlp_f1, lstm_f1):.4f}). "
        f"The MLP misclassified {results['misclassified_test_rows']['Feed-forward MLP']} "
        f"of {results['test_rows']:,} test rows, and the LSTM misclassified "
        f"{results['misclassified_test_rows']['PyTorch LSTM']}. The majority "
        "baseline is included as a reference floor, not as a competitive model."
    )
    weakest_intents = sorted(
        results["per_intent_metrics"]["Feed-forward MLP"].items(),
        key=lambda item: item[1]["f1-score"],
    )[:5]
    document.add_paragraph(
        "The five lowest per-intent MLP F1 values on this test run were: "
        + "; ".join(
            f"{name} ({values['f1-score']:.3f}, n={int(values['support'])})"
            for name, values in weakest_intents
        )
        + ". This class-level view complements the macro average."
    )
    document.add_paragraph(
        "Figure 3 plots micro-averaged one-vs-rest ROC curves, whereas the table "
        "reports macro one-vs-rest ROC-AUC. These are different averaging "
        "conventions and are labeled separately. The ROC-AUC values near one "
        "indicate strong ranking on this dataset split; they do not remove the "
        "synthetic-data or duplicate-similarity limitations."
    )

    document.add_heading("Visual Results and Training Evidence", level=1)
    document.add_paragraph(
        "The following figures are the actual PNG outputs saved by the executed "
        "notebook. They provide visual evidence of the held-out confusion "
        "matrices, ROC curves, and training-loss histories."
    )
    add_figure(
        document,
        IMAGE_DIR / "confusion_matrices.png",
        "Figure 2. Row-normalized MLP and LSTM confusion matrices on the held-out test partition.",
    )
    add_figure(
        document,
        IMAGE_DIR / "roc_curves.png",
        "Figure 3. Micro-averaged one-vs-rest ROC curves on the held-out test partition.",
    )
    add_figure(
        document,
        IMAGE_DIR / "training_loss.png",
        "Figure 4. Training cross-entropy loss by epoch for the MLP and LSTM.",
    )
    document.add_paragraph(
        f"Validation macro F1 selected epoch {mlp_best_epoch} of "
        f"{len(mlp_history)} for the MLP and epoch {lstm_best_epoch} of "
        f"{len(lstm_history)} for the LSTM. The saved loss plots show training "
        "loss; validation macro-F1 histories are retained in the JSON results."
    )

    document.add_heading("Limitations and Conclusion", level=1)
    document.add_paragraph(
        "The source dataset is synthetic/hybrid and automatically labeled. A "
        "stratified group split prevents the same case- and whitespace-normalized "
        "request from appearing in multiple partitions, but does not rule out "
        "overlap among paraphrases or generated templates. The test results "
        "therefore measure performance on this held-out portion of the source "
        "corpus, not on independently collected customer traffic. No operational "
        "claims about routing accuracy, queue performance, or service quality "
        "should be inferred without external validation."
    )
    document.add_paragraph(
        "The project preserves the existing Bitext dataset, 27-intent target, "
        "TF-IDF MLP, and PyTorch LSTM. It corrects the evaluation procedure for "
        "repeated request text, records the resulting test metrics, and provides "
        "the analysis and visual evidence needed to interpret the models."
    )

    document.add_heading("Repository and Reproducibility", level=1)
    repository_paragraph = document.add_paragraph()
    repository_paragraph.add_run("GitHub repository: ").bold = True
    add_hyperlink(repository_paragraph, GITHUB_URL, GITHUB_URL)
    document.add_paragraph(
        "The executed notebook, saved metrics, report builder, and figure PNGs "
        "are in the repository's `week 6/` directory. From the repository root, "
        "install the Week 6 requirements and open the notebook with that folder "
        "as the working directory. Re-running the notebook downloads the "
        "published dataset if it is not already present in the user's cache, "
        "trains both models, and regenerates the figures and JSON metrics."
    )
    document.add_paragraph(
        "To rebuild this report after executing the notebook, run "
        "`python \"week 6/build_week7_report.py\"` from the repository root."
    )

    document.add_heading("References", level=1)
    document.add_paragraph(
        "Bitext. (n.d.). *Bitext customer support LLM chatbot training dataset* "
        "[Data set]. Hugging Face. "
        "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )
    document.add_paragraph(
        "Fawcett, T. (2006). An introduction to ROC analysis. "
        "*Pattern Recognition Letters, 27*(8), 861–874. "
        "https://doi.org/10.1016/j.patrec.2005.10.010"
    )
    document.add_paragraph(
        "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
        "*Neural Computation, 9*(8), 1735–1780. "
        "https://doi.org/10.1162/neco.1997.9.8.1735"
    )
    document.add_paragraph(
        "Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic "
        "text retrieval. *Information Processing & Management, 24*(5), 513–523. "
        "https://doi.org/10.1016/0306-4573(88)90021-0"
    )
    document.add_paragraph(
        "Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance "
        "measures for classification tasks. *Information Processing & Management, "
        "45*(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002"
    )

    document.save(REPORT_PATH)
    shutil.copyfile(REPORT_PATH, CANONICAL_REPORT_PATH)
    print(f"Saved technical report: {REPORT_PATH}")
    print(f"Updated canonical Week 6 report: {CANONICAL_REPORT_PATH}")
    print(f"Test rows: {results['test_rows']:,}; MLP macro F1: {mlp_f1:.4f}; LSTM macro F1: {lstm_f1:.4f}")


if __name__ == "__main__":
    main()
