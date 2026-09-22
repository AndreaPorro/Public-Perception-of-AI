# 🧠 Public Perception of AI: A Computational Analysis of the Reddit Debate

This repository contains the bachelor's thesis project developed for the **BSc in Statistics and Information Management** at the **University of Milan-Bicocca (UNIMIB)**.

The project investigates the public perception of Artificial Intelligence through a computational pipeline applied to over 50,000 Reddit comments: from data collection via the Reddit API to **Sentiment Analysis**, **Emotion Analysis**, **Topic Modeling**, and **Explainable AI (XAI)**.

## 🎯 Project Objectives & Research Questions

The analysis is structured around three core research questions, each addressing a different layer of the public debate on AI:

* **What are the dominant themes in the conversation?** Mapping the topics that structure the public discourse on AI, from workforce impact to global governance, through unsupervised topic modeling.
* **What emotions does each theme provoke?** Going beyond simple positive/negative polarity to identify the specific emotions (anger, fear, joy, sadness...) associated with each topic.
* **What makes an opinion more influential than another?** Using Explainable AI to uncover which factors — contextual, textual, or emotional — actually drive a comment's engagement score.


## 🛠️ Methodology

The project follows an end-to-end computational pipeline, combining classical NLP preprocessing with modern Transformer-based models:

* **Data Collection:** ~57,900 comments and 250 posts collected from **r/Futurology** via the Reddit API (`PRAW`), filtered across 5 thematic queries (AGI, regulation, labor impact, safety, ethics).
* **Text Preprocessing:** Cleaning, lowercasing, punctuation/stopword removal via `spaCy`, followed by exploratory data analysis and text mining (frequencies, bigrams, trigrams).
* **Sentiment Analysis:** Benchmark of 8 Hugging Face models against a labeled Kaggle dataset, followed by a weighted ensemble of the 4 best performers, validated against a hand-labeled sample.
* **Emotion Analysis:** Fine-grained emotion classification (anger, disgust, fear, joy, sadness, surprise, neutral) via the `j-hartmann/emotion-english-distilroberta-base` model.
* **Topic Modeling:** Unsupervised topic discovery on post titles using `BERTopic`, identifying 7 distinct themes in the debate.
* **Explainable AI (XAI):** A Random Forest classifier predicting comment engagement (low/medium/high), interpreted via `SHAP` to uncover the key drivers of influence.

## 📂 Repository Structure

**`notebooks/`** — Core analysis pipeline

- `Download_reddits.ipynb` — Data collection from r/Futurology via the Reddit API
- `Reddit_analysis.ipynb` — Main notebook: cleaning, EDA, text mining, sentiment/emotion analysis, topic modeling, and the XAI predictive model

**`src/`** — Reusable Python modules imported by the notebooks

- `sentiment_analysis_modular.py` — Ensemble sentiment analysis (4 Hugging Face models)
- `emotion_recognition.py` — Emotion classification (j-hartmann model)

**`model_selection/`** — Model benchmarking and validation

- `kaggle_benchmark/` — Comparison of 8 sentiment/emotion models on a labeled Kaggle dataset
- `manual_validation/` — Validation against a 100-comment hand-labeled sample, including the saved BERT+Logistic Regression classifier

**`data/`** — Raw datasets

- `Reddit_Post_ChatGPT.csv` — Raw posts collected from Reddit
- `Reddit_Comments_ChatGPT.csv` — Raw comments collected from Reddit

**`report/`** — Final deliverables

- `thesis_report.pdf` — Full thesis document
- `thesis_slides.pptx` — Presentation slides

## 📊 Key Insights

* **The sentiment ensemble reached 70% accuracy** on a hand-labeled validation sample, a major improvement over the best single model (54%) and the BERT+ML approach (36%).
* **Negative sentiment dominates the debate** (21,896 comments) but not overwhelmingly — positive and neutral sentiment together make up more than half of the dataset, suggesting a nuanced rather than polarized conversation.
* **Anger is the most frequent non-neutral emotion** (16.3% overall), peaking at **42.6%** in discussions about AI regulation and governance — the most emotionally polarized topic in the dataset.
* **Contextual metrics outweigh emotional content** in predicting a comment's engagement: `Num_Comments` and `score_post` are the strongest SHAP predictors, more influential than the comment's own sentiment or emotion.
* **Disgust and anger are the most "efficient" emotions** at generating engagement, while fear — counterintuitively — is the least efficient, despite being the most emotionally intense (highest average confidence score).

## 📄 Report & Slides

* **[Full Thesis Report](./report/thesis_report.pdf)** — complete methodology, literature review, and discussion of results
* **[Presentation Slides](./report/thesis_slides.pptx)** — summary deck used for the thesis defense

## 👤 Author

*Developed by **Andrea Porro** as part of the BSc in Statistics and Information Management program at the **University of Milan-Bicocca (UNIMIB)**.*
