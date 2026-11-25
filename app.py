from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import DATA_DIR, MODEL_DIRS, TASK_DISPLAY_NAMES, TASK_LABELS
from src.demo import demo_predict
from src.predict import ModelNotFoundError, model_available, predict_many

st.set_page_config(
    page_title="Spam & Hate Speech AI Lab",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1440px; padding-top: 1.7rem; padding-bottom: 3rem;}
.hero {padding: 1.45rem 1.6rem; border: 1px solid #dbe5f2; border-radius: 18px;
       background: linear-gradient(115deg, #eef5ff 0%, #ffffff 60%, #effaf7 100%);}
.hero h1 {margin: 0 0 .3rem 0; color: #132a46; letter-spacing: -.035em;}
.hero p {margin: 0; color: #52657c; font-size: 1.02rem;}
.metric-card {border: 1px solid #e1e8f0; border-radius: 14px; padding: 1rem 1.1rem; background: white; min-height: 118px;}
.metric-label {font-size: .82rem; color: #62738a; text-transform: uppercase; letter-spacing: .06em;}
.metric-value {font-size: 1.35rem; font-weight: 700; color: #17365d; margin-top: .35rem;}
.small-muted {color: #68788c; font-size: .86rem;}
.stTabs [data-baseweb="tab-list"] {gap: .45rem;}
.stTabs [data-baseweb="tab"] {border-radius: 9px 9px 0 0; padding: .65rem 1rem;}
</style>
<div class="hero">
  <h1>🛡️ Spam & Hate Speech Detection AI Lab</h1>
  <p>Fine-tuned transformer models, transparent evaluation, and hands-on testing for text moderation.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Model controls")
    spam_ready = model_available("spam")
    hate_ready = model_available("hate")
    st.write("**Spam model**", "✅ Trained model ready" if spam_ready else "⚪ Not trained")
    st.write("**Hate model**", "✅ Trained model ready" if hate_ready else "⚪ Not trained")
    demo_enabled = st.toggle(
        "Allow demo rules when model is missing",
        value=True,
        help="Demo rules only preview the workflow. They are not machine-learning predictions and must not be used for moderation decisions.",
    )
    st.divider()
    st.caption("Decision thresholds")
    spam_threshold = st.slider("Spam threshold", 0.05, 0.95, 0.50, 0.05)
    hate_threshold = st.slider("Hate threshold", 0.05, 0.95, 0.50, 0.05)
    st.divider()
    st.caption("Training commands")
    st.code("python -m src.train --task all --epochs 3", language="bash")
    st.caption("Then restart / refresh the app if model status does not update.")


def get_predictions(task: str, texts: list[str], threshold: float, use_demo: bool) -> list[dict]:
    """Predict valid rows and preserve empty rows as explicit input errors."""
    normalized = [" ".join(str(text or "").split()) for text in texts]
    valid_indices = [index for index, text in enumerate(normalized) if text]
    results = [None] * len(normalized)
    for index, text in enumerate(normalized):
        if not text:
            results[index] = {
                "text": "", "label_id": -1, "label": "invalid/empty",
                "positive_probability": None, "confidence": None,
                "class_probabilities": {}, "threshold": threshold,
                "source": "input validation",
            }
    if not valid_indices:
        return results
    valid_texts = [normalized[index] for index in valid_indices]
    try:
        valid_results = predict_many(task, valid_texts, threshold=threshold)
    except ModelNotFoundError:
        if not use_demo:
            raise
        valid_results = [demo_predict(task, text, threshold=threshold) for text in valid_texts]
    for index, result in zip(valid_indices, valid_results):
        results[index] = result
    return results


def render_result(task: str, result: dict):
    positive_label = TASK_LABELS[task][1]
    negative_label = TASK_LABELS[task][0]
    positive = result["label_id"] == 1
    title = f"{result['label'].upper()}"
    st.markdown(f"### {'🚨' if positive else '✅'} {title}")
    st.caption(f"Prediction source: {result['source']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted class", result["label"])
    c2.metric("Model confidence", f"{result['confidence']:.1%}")
    c3.metric(f"P({positive_label})", f"{result['positive_probability']:.1%}")
    st.progress(result["positive_probability"], text=f"Positive-class score: {result['positive_probability']:.1%}")
    with st.expander("View both class probabilities"):
        st.write({negative_label: f"{result['class_probabilities'][negative_label]:.1%}",
                  positive_label: f"{result['class_probabilities'][positive_label]:.1%}"})


def run_task(task: str, texts: list[str], threshold: float) -> list[dict]:
    return get_predictions(task, texts, threshold, demo_enabled)


single_tab, batch_tab, scenarios_tab, guide_tab = st.tabs([
    "✍️ Single message", "📁 Batch CSV", "🧪 Scenario lab", "📘 Guide & model status"
])

with single_tab:
    st.subheader("Test a message")
    st.write("Run either classifier or both classifiers against the same text.")
    selected_task = st.radio(
        "Classifier", ["Both", "Spam only", "Hate speech only"], horizontal=True,
    )
    default_text = "Congratulations! You have won a free reward. Click here to claim it: https://example.com"
    message = st.text_area("Message to analyze", value=default_text, height=150,
                           placeholder="Paste an SMS, email, comment, or social media post...")
    analyze = st.button("Analyze message", type="primary", use_container_width=False)
    if analyze:
        if not message.strip():
            st.warning("Enter a message before running the analysis.")
        else:
            tasks = ["spam", "hate"] if selected_task == "Both" else (["spam"] if selected_task == "Spam only" else ["hate"])
            columns = st.columns(len(tasks))
            for column, task in zip(columns, tasks):
                with column:
                    st.markdown(f"#### {TASK_DISPLAY_NAMES[task]}")
                    threshold = spam_threshold if task == "spam" else hate_threshold
                    try:
                        render_result(task, run_task(task, [message], threshold)[0])
                    except ModelNotFoundError as exc:
                        st.error(str(exc))

with batch_tab:
    st.subheader("Analyze a CSV file")
    st.write("Upload a CSV containing a column named **text**. The app adds predicted labels and scores.")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"], key="batch_csv")
    batch_tasks = st.multiselect("Classifiers to run", ["spam", "hate"], default=["spam", "hate"])
    if uploaded is not None:
        try:
            source_df = pd.read_csv(uploaded)
            if "text" not in source_df.columns:
                st.error("Your CSV needs a column named 'text'.")
            elif source_df.empty:
                st.warning("The uploaded CSV has no rows.")
            else:
                st.caption(f"Loaded {len(source_df):,} rows")
                st.dataframe(source_df.head(10), use_container_width=True, hide_index=True)
                if st.button("Run batch analysis", type="primary"):
                    if not batch_tasks:
                        st.warning("Select at least one classifier.")
                    else:
                        output_df = source_df.copy()
                        for task in batch_tasks:
                            threshold = spam_threshold if task == "spam" else hate_threshold
                            try:
                                results = get_predictions(task, source_df["text"].fillna("").astype(str).tolist(), threshold, demo_enabled)
                                output_df[f"{task}_prediction"] = [r["label"] for r in results]
                                output_df[f"{task}_positive_score"] = [round(r["positive_probability"], 4) if r["positive_probability"] is not None else None for r in results]
                                output_df[f"{task}_source"] = [r["source"] for r in results]
                            except ModelNotFoundError as exc:
                                st.error(str(exc))
                                break
                        else:
                            st.success("Batch analysis complete.")
                            st.dataframe(output_df, use_container_width=True, hide_index=True)
                            st.download_button(
                                "Download predictions CSV",
                                data=output_df.to_csv(index=False).encode("utf-8"),
                                file_name="spam_hate_predictions.csv",
                                mime="text/csv",
                            )
        except Exception as exc:
            st.error(f"Could not read this CSV: {exc}")
    st.download_button(
        "Download sample CSV template",
        data="text\n\"Your parcel is arriving tomorrow.\"\n\"URGENT: claim your free reward at https://example.com\"\n".encode("utf-8"),
        file_name="sample_batch_template.csv", mime="text/csv",
    )

with scenarios_tab:
    st.subheader("Scenario testing lab")
    st.write("Use the bundled examples to probe routine, suspicious, borderline, and context-sensitive messages.")
    sample_path = DATA_DIR / "sample_cases.csv"
    cases = pd.read_csv(sample_path)
    category_options = ["All"] + sorted(cases["scenario"].dropna().unique().tolist())
    category = st.selectbox("Filter scenarios", category_options)
    visible = cases if category == "All" else cases[cases["scenario"] == category]
    st.dataframe(visible[["scenario", "text", "expected_spam", "expected_hate", "notes"]],
                 use_container_width=True, hide_index=True)
    scenario_tasks = st.multiselect("Run classifiers", ["spam", "hate"], default=["spam", "hate"], key="scenario_tasks")
    if st.button("Run selected scenario tests", type="primary"):
        outputs = visible.copy()
        try:
            for task in scenario_tasks:
                threshold = spam_threshold if task == "spam" else hate_threshold
                results = get_predictions(task, visible["text"].astype(str).tolist(), threshold, demo_enabled)
                outputs[f"predicted_{task}"] = [r["label"] for r in results]
                outputs[f"{task}_positive_score"] = [round(r["positive_probability"], 3) if r["positive_probability"] is not None else None for r in results]
                outputs[f"{task}_prediction_source"] = [r["source"] for r in results]
            st.success("Scenario run completed. Review borderline cases manually.")
            st.dataframe(outputs, use_container_width=True, hide_index=True)
            st.download_button(
                "Download scenario results",
                data=outputs.to_csv(index=False).encode("utf-8"),
                file_name="scenario_test_results.csv", mime="text/csv",
            )
        except ModelNotFoundError as exc:
            st.error(str(exc))

with guide_tab:
    st.subheader("Model readiness")
    for task in ("spam", "hate"):
        exists = model_available(task)
        with st.expander(f"{TASK_DISPLAY_NAMES[task]} — {'ready' if exists else 'not trained'}", expanded=not exists):
            st.write(f"Model folder: `{MODEL_DIRS[task]}`")
            if exists:
                metrics_path = MODEL_DIRS[task] / "evaluation_metrics.json"
                st.success("A fine-tuned model is available locally.")
                if metrics_path.exists():
                    try:
                        import json
                        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                        cols = st.columns(4)
                        for col, key in zip(cols, ["accuracy", "precision", "recall", "f1"]):
                            col.metric(key.replace("_", " ").title(), f"{metrics.get(key, 0):.3f}")
                        matrix = metrics.get("confusion_matrix")
                        if matrix:
                            st.caption("Confusion matrix (rows = actual, columns = predicted)")
                            st.dataframe(
                                pd.DataFrame(matrix, index=[TASK_LABELS[task][0], TASK_LABELS[task][1]],
                                             columns=[TASK_LABELS[task][0], TASK_LABELS[task][1]]),
                                use_container_width=True,
                            )
                        if metrics.get("classification_report"):
                            with st.expander("View class-wise evaluation report"):
                                st.json(metrics["classification_report"])
                    except Exception:
                        st.info("Model metrics file could not be displayed.")
            else:
                st.info(f"Train this model with `python -m src.train --task {task} --epochs 3`.")
    st.divider()
    st.markdown("""
**Datasets**

- Spam: UCI SMS Spam Collection, loaded through `ucirvine/sms_spam`.
- Hate speech: CardiffNLP TweetEval, `hate` configuration.

**Important:** spam and hate speech are different classification tasks. A message may be both, neither, or ambiguous. The hate model identifies patterns in its English Twitter training data; it does not determine intent, legality, or a person's beliefs. Context, quotation, sarcasm, dialect, and reclaimed language can change interpretation.

**Demo mode:** when a fine-tuned model is missing, optional hand-written rules let you explore the interface. Demo results are not AI model outputs and are not suitable for real moderation decisions.
""")

st.divider()
st.markdown('<p class="small-muted">Educational prototype • English-language datasets • Human review recommended for consequential moderation decisions</p>', unsafe_allow_html=True)
