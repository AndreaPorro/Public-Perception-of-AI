"""
Module for sentiment analysis using an ensemble of 4 models.
"""

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch
import numpy as np
import pandas as pd


def analizza_sentimento(pipe, text):
    """
    Basic function for analyzing the sentiment of a text.
    
    Args:
        pipe: Hugging Face pipeline
        text: text to be analyzed
    
    Returns:
        sentiment label or None in case of error
    """

    try:
        result = pipe(text[:512])[0]  
        return result['label']
    except Exception as e:
        print(f"Error in analysis: {e}")
        return None


def carica_modello_1():
    """Upload the first model: cardiffnlp/twitter-roberta-base-sentiment-latest"""

    print("Loading model 1...")
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    mapping = {
        'neutral': 0,
        'positive': 1,
        'negative': -1
    }
    return pipe, mapping


def carica_modello_5():
    """Upload the second model: cardiffnlp/twitter-roberta-base-sentiment"""

    print("Loading model 5...")
    pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
    mapping = {
        'LABEL_2': 1,
        'LABEL_0': -1,
        'LABEL_1': 0
    }
    return pipe, mapping


def carica_modello_7():
    """Upload the third model: akshataupadhye/finetuning-sentiment-model-reddit-data"""

    print("Loading model 7...")
    pipe = pipeline("text-classification", model="akshataupadhye/finetuning-sentiment-model-reddit-data")
    mapping = {
        'LABEL_2': 1,
        'LABEL_0': -1,
        'LABEL_1': 0
    }
    return pipe, mapping


def carica_modello_chelberta():
    """Upload the fourth model: Chelberta with PEFT"""

    print("Loading model Chelberta...")

    base_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
    
    model = PeftModel.from_pretrained(base_model, "UAlbertaUAIS/Chelberta")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    mapping = {
        'positive': 1,
        'negative': -1,
        'neutral': 0
    }
    
    return pipe, mapping


def aggiungi_predizioni_singolo_modello(df, colonna_testo, pipe, mapping, nome_colonna_risultato):
    """
    Adds a column containing the predictions from a single model.
    
    Args:
        df: pandas DataFrame
        colonna_testo: name of the column containing the texts
        pipe: Hugging Face pipeline
        mapping: dictionary to map labels to numerical values
        nome_colonna_risultato: name of the new column to create

    Returns:
        DataFrame with the new column added
    """

    print(f"Analysis in progress for {nome_colonna_risultato}...")
    df[nome_colonna_risultato] = df[colonna_testo].apply(lambda x: analizza_sentimento(pipe, x))
    df[nome_colonna_risultato] = df[nome_colonna_risultato].replace(mapping)
    return df


def calcola_ensemble(df, colonne_predizioni, accuracies):
    """
    Calculates the final sentiment using weighted ensemble.

    Args:
        df: DataFrame with the predictions of individual models
        colonne_predizioni: list of column names with predictions
        accuracies: dictionary with the accuracies of each model
    
    Returns:
        DataFrame with the 'Final_ensemble' column added
    """
    
    print("Calculating final ensemble...")
    
    total = sum(accuracies.values())
    weights = {k: v / total for k, v in accuracies.items()}
    
    weighted_sum = np.zeros(len(df))
    for col, w in weights.items():
        weighted_sum += df[col] * w
    
    def decision(score):
        if score > 0.25:
            return 1
        elif score < -0.25:
            return -1
        else:
            return 0
    
    df['Final_ensemble'] = [decision(s) for s in weighted_sum]
    
    return df


def analizza_sentiment_completo(df, colonna_testo='clean_comment'):
    """
    MAIN FUNCTION: Performs a comprehensive sentiment analysis using all four models.
    
    Args:
        df: pandas DataFrame with the texts to analyze
        colonna_testo: name of the column containing the texts (default: 'clean_comment')

    Returns:
        DataFrame with 5 new columns: Ris_1, Ris_5, Ris_7, Ris_Chelberta, Final_ensemble
    """
    
    pipe1, mapping1 = carica_modello_1()
    pipe5, mapping5 = carica_modello_5()
    pipe7, mapping7 = carica_modello_7()
    pipe_chel, mapping_chel = carica_modello_chelberta()
    
    df = aggiungi_predizioni_singolo_modello(df, colonna_testo, pipe1, mapping1, 'Ris_1')
    df = aggiungi_predizioni_singolo_modello(df, colonna_testo, pipe5, mapping5, 'Ris_5')
    df = aggiungi_predizioni_singolo_modello(df, colonna_testo, pipe7, mapping7, 'Ris_7')
    df = aggiungi_predizioni_singolo_modello(df, colonna_testo, pipe_chel, mapping_chel, 'Ris_Chelberta')
    
    accuracies = {
        'Ris_1': 49.61,
        'Ris_5': 50.34,
        'Ris_7': 75.94,
        'Ris_Chelberta': 51.73
    }
    
    df = calcola_ensemble(df, ['Ris_1', 'Ris_5', 'Ris_7', 'Ris_Chelberta'], accuracies)

    print("✅ Analysis completed!")
    return df