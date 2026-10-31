# Routing Before Querying

**Text Features and LLM Configuration Selection in Hate-Speech Annotation: A Pilot Study**

Fatemeh Mohammadi, Paolo Ceravolo
Department of Computer Science, Università degli Studi di Milano

This repository contains the code, data outputs, and figures for the paper *Routing Before Querying*. The paper asks whether simple, interpretable features of a text can indicate, **before any model is queried**, which LLM–prompt configuration is most likely to annotate it correctly.

## Overview

LLM committees and LLM chains are increasingly used to support human annotators in hate-speech labelling. Committees multiply query costs, and chains require choosing a model order and an escalation threshold. We study a cheaper alternative: **routing each instance to the LLM–prompt configuration best suited to its textual and linguistic features**, so that each instance costs exactly one query.

In this pilot study we use 200 human-labelled Italian instances from the *Mappa dell'Intolleranza* project, annotated with four classes (*neutral*, *hate speech*, *mixed*, *stereotype*). We compare six configurations against 44 extracted text features.

| Model | Prompt strategies |
|---|---|
| `Qwen/Qwen2.5-1.5B-Instruct` | zero-shot (ZS), few-shot (FS), definition + few-shot (Def+FS) |
| `google/gemma-3-1b-it` | zero-shot (ZS), few-shot (FS), definition + few-shot (Def+FS) |

## Key findings

- **Similar aggregate scores.** Macro-F1 across the six configurations differs by only 2.8 points (0.277 to 0.305).
- **Large differences by class.** Within a single configuration, per-class F1 differs by up to 73.2 points. The *mixed* and *stereotype* classes depend most on the configuration.
- **The best configuration depends on the features.** For 26 of 38 usable features, the top configuration changes across feature bins.
- **Correct answers are spread across configurations.** An oracle that picks the best configuration per instance labels 70.0% of instances correctly. The best single configuration reaches 54.3%, the majority-class baseline 50.5%, and a uniformly random router 51.1%.
- **Cost differs more than quality.** Mean latency ranges from 0.405 s (Gemma ZS) to 0.956 s (Qwen Def+FS), roughly 2.3× apart in macro-F1 per second.

These results do not yet establish a working router. They test the premise a router needs: that configurations behave differently in different regions of the feature space.

## Repository structure

```
RoutingBeforeQuerying/
├── code/
│   └── feature-prediction-relationship-mappa.ipynb   # Full pipeline (Kaggle notebook)
├── input/
│   └── sample_4_annotation_seg2_done.xlsx            # 200 human-annotated instances
├── prediction/
│   └── label_predictions.xlsx                        # Predictions of all six configurations
├── output_files/
│   ├── enriched_dataset.xlsx         # Instances + 44 extracted features
│   ├── feature_eda_summary.csv       # Descriptive statistics, skewness, sparsity
│   ├── binary_feature_summary.csv    # Summary of the binary features
│   ├── feature_correlations.csv      # Pairwise Pearson correlations
│   ├── bin_performance.xlsx          # Macro-F1 of every configuration in every feature bin
│   ├── best_by_bin.xlsx              # Best configuration per feature bin
│   ├── best_by_bin_reliable.xlsx     # Same, restricted to bins with enough instances
│   ├── feature_winners.xlsx          # Winning configurations per feature
│   └── routing_signal.xlsx           # Features whose best configuration changes across bins
├── figures/
│   ├── features_dist.png                             # Distributions of the numerical features (Fig. 1)
│   ├── DistributionOfNumericalFeatures.pdf
│   ├── CorrelationMatrixHighestVarianceFeatures.pdf  # Correlation matrix (Fig. 2)
│   ├── confusion_matrix_from_label_results.pdf       # Pooled confusion matrix (Fig. 3)
│   ├── label_performance_latency.pdf                 # Macro-F1 vs. latency (Fig. 4)
│   ├── label_configuration_performance.pdf           # Per-configuration performance
│   ├── PCACumulativeExplainedVariance.pdf
│   ├── PairwiseScatterPlots.pdf
│   └── feature_routing_plots/                        # Per-bin macro-F1 of all configurations, one plot per feature
└── Related work/                                     # Reference material for the related-work section
```

## Pipeline

The notebook `code/feature-prediction-relationship-mappa.ipynb` runs the whole study end to end:

1. **Data loading and checks.** Reads the annotated file (columns `Clean_Text`, `Annotated_Label`) and validates the four labels.
2. **Feature extraction.** Computes 44 features in 12 groups with spaCy's Italian pipeline (`it_core_news_sm`), curated Italian lexicons, and the `MilaNLProc/feel-it-italian-sentiment` model. The groups are: text statistics, lexical, readability (Gulpease, Flesch–Vacca), syntactic complexity, social-media markers, sentiment, offensive language, pragmatics, emphasis and style, POS distribution, named entities, and sarcasm proxies.
3. **Exploratory data analysis.** Distributions, skewness (adjusted Fisher–Pearson coefficient), sparsity, correlations, and PCA.
4. **LLM annotation.** Runs the six model–prompt configurations locally with Hugging Face Transformers and deterministic decoding. Outputs that do not match a valid label are kept and reported as `INVALID`.
5. **Evaluation.** Accuracy, macro-F1, per-class F1, latency, token counts, and invalid-output rate per configuration.
6. **Feature-conditioned analysis.** Splits each feature into quartile bins (`pd.qcut`, 4 bins), finds the best configuration in each bin, and measures how often the winner changes.
7. **Complementarity analysis.** Counts how many configurations solve each instance, and computes the oracle and unique-solve statistics.

## Reproducing the results

The notebook was developed on **Kaggle** with a GPU.

To run outside Kaggle, replace the `kaggle_secrets` call in `load_generators()` with `os.environ["HF_TOKEN"]` and update the input and output paths.

### Requirements

```bash
pip install torch transformers pandas numpy scikit-learn scipy spacy emoji matplotlib seaborn tqdm openpyxl
python -m spacy download it_core_news_sm
```

The random seed is fixed at `RANDOM_SEED = 42`. It controls the selection of few-shot examples (one per class) and the other sampling steps.

## Citation

If you use this code or the analysis, please cite:

```bibtex
@inproceedings{mohammadi2026routing,
  title     = {Routing Before Querying: Text Features and LLM Configuration Selection in Hate-Speech Annotation -- A Pilot Study},
  author    = {Mohammadi, Fatemeh and Ceravolo, Paolo},
  booktitle = {The 4th International Conference on Foundation and Large Language Models (FLLM2026)},
  year      = {2026}
}
```

## Related work from our group

- F. Mohammadi, S. Maghool, P. Ceravolo. *From Annotation to Detection: Evaluating LLMs for Hate Speech and Stereotype Identification in Italian.* IEEE/ACS AICCSA 2025. [doi:10.1109/AICCSA66935.2025.11315321](https://doi.org/10.1109/aiccsa66935.2025.11315321)
- F. Mohammadi, P. Ceravolo, S. Maghool, M. A. Tamborini, M. E. D'Amico. *EHLD: A General Data Framework for Harmful Language Detection and Evaluation.* Digital Library Perspectives, 2026. [doi:10.1108/DLP-01-2026-0028](https://doi.org/10.1108/DLP-01-2026-0028)
- F. Mohammadi, T. Romano, S. Maghool, P. Ceravolo. *Artificial Conversations, Real Results: Fostering Language Detection with Synthetic Data.* TRUST-AI 2025, CEUR-WS vol. 4132. [PDF](https://ceur-ws.org/Vol-4132/short45.pdf)

## Contact

Fatemeh Mohammadi, fatemeh.mohammadi@unimi.it
