| Model | Family | ISOT F1 | COVID F1 | LIAR F1 | LIAR-6 Acc | LIAR-6 F1 |
|---|---|---|---|---|---|---|
| TF-IDF + LR (baseline) | Baseline | 0.980 | 0.929 | 0.592 | 0.252 | 0.219 |
| bow-nb | Traditional ML | – | – | 0.612 | – | – |
| glove-lr | Traditional ML | 0.962 | 0.880 | 0.589 | 0.270 | 0.248 |
| tfidf-lr | Traditional ML | – | – | – | 0.243 | 0.230 |
| tfidf-svm | Traditional ML | 0.990 | 0.948 | – | – | – |
| BiLSTM | Sequential DL | 0.996 | 0.951 | 0.595 | 0.246 | 0.240 |
| gat | Graph | 0.825 | 0.854 | 0.366 | 0.198 | 0.061 |
| textgcn | Graph | 0.962 | 0.924 | 0.606 | 0.221 | 0.195 |
| bert | Transformer | **0.999** | 0.973 | 0.627 | 0.287 | **0.286** |
| distilbert | Transformer | 0.998 | 0.974 | **0.646** | 0.291 | 0.277 |
| roberta | Transformer | 0.998 | **0.976** | 0.639 | **0.298** | 0.280 |
| llm-zero-shot | LLM | 0.310 | 0.327 | 0.299 | 0.206 | 0.087 |

Test set, macro-F1 unless marked Acc (accuracy). **Bold** = best per column.