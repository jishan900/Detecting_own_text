# Detecting_own_text


---

# Do LLMs Recognize Their Own Writing?

This project investigates whether Large Language Models (LLMs), such as GPT-2, can recognize their own generated text based solely on **internal activation patterns**. We extract hidden state vectors from an LLM and train a classifier to distinguish between **human-authored** and **LLM-generated** text.

---

## Project Structure

```

LLM\_Self\_Recognition.ipynb
Dataset/
│   ├── c4-shard-00000.json.gz              
│   ├── c4\_prompt\_completion\_pairs.csv      
│   ├── gpt2\_prompt\_completion\_pairs.csv    
│   ├── combined\_prompt\_completion\_pairs.csv
│   ├── activations\_dataset.csv             

Visuals/
│   ├── confusion\_matrix.png
│   ├── roc\_curve.png
│   ├── pr\_curve.png
│   └── top\_20\_features.png

````

---

## Objectives

- Extract activations from a pretrained LLM (GPT-2).
- Compare **human-written** vs. **GPT-2-generated** completions.
- Train a classifier (e.g., XGBoost) to distinguish them using activation patterns alone.
- Visualize results and explore feature importance.
- Evaluate generalizability across models or datasets.

---

## Methodology Overview

### 1. **Data Collection**
- Used the `C4-realnewslike` dataset (a high-quality subset of C4).
- Extracted 18,000 prompt-completion pairs of human-written text.
- Generated matching GPT-2 completions using `distilgpt2`.

### 2. **Activation Extraction**
- Passed each completion through `distilgpt2`.
- Extracted the **last hidden layer's activation** for the **final token**.


### 3. **Classification**
- Used **XGBoost** to classify activation vectors as:
  - `human` (label = 0)
  - `gpt2` (label = 1)
- Trained/tested on a combined dataset of 36,000 samples (balanced).
- Visualized performance using ROC, PR curve, and feature importance.

---

## Results Summary

- **Accuracy:** 93%
- **Precision (GPT-2):** 96%
- **Recall (GPT-2):** 90%
- **F1-Score (macro avg):** 0.93
- **ROC AUC:** 0.98
- **Average Precision:** 0.98

These metrics demonstrate that the classifier can **effectively distinguish** between LLM and human text purely from internal activations.

| Confusion Matrix | ROC Curve | PR Curve | Feature Importance |
|------------------|-----------|----------|---------------------|
| ![Confusion Matrix](https://github.com/jishan900/Detecting_own_text/blob/master/Plots/confusion_matrix.png) | ![ROC](Visuals/roc_curve.png) | ![PR](Visuals/pr_curve.png) | ![Features](Visuals/top_20_features.png) |

---

## Generalization Strategy

To validate robustness:
- **Cross-model**: Train on `distilgpt2`, test on completions from `gpt2-large`, `mistral`, `llama`.
- **Cross-dataset**: Use prompts from Reddit, Wikipedia, or scientific texts.
- **Temporal**: Train on older articles, test on recent writing.
- **Few-shot**: Train on limited samples and generalize to unseen prompts.

---

## Future Improvements

- Extract pooled or multi-layer embeddings instead of final-token only.
- Use larger LLMs (e.g., `llama-3`, `mistral`) for more expressive activation spaces.
- Apply deep classifiers like MLPs.
- Explore explainability with SHAP/LIME.

---

## Dependencies

```bash
transformers
tensorflow
xgboost
scikit-learn
pandas
matplotlib
seaborn
````

Install with:

```bash
pip install transformers tensorflow xgboost scikit-learn pandas matplotlib seaborn
```

---

## Credits

This project was developed as part of a research investigation into LLM self-recognition, activation analysis, and classification of AI-generated content. Dataset subset from AllenAI and Hugging Face.

---

## License

This project is for academic and research use. Dataset licenses are inherited from the original `C4` and `GPT-2` usage terms.

---


