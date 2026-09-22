"""
Emotion Analysis Function - Use of the j-hartmann model on Hugging Face
"""

import pandas as pd
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


def carica_modello_emotion():
    """Upload the model: j-hartmann (social media)"""

    print("Loading emotion model (j-hartmann)...")
    pipe = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )
    return pipe


def analizza_emozione(text, pipe):
    """
    Analyze the sentiment of a single text.
    
    Args:
        text: text to be analyzed
        pipe: pipeline of the j-hartmann model

    Returns:
        tuple: (emotion, score)
    """
    
    if pd.isna(text) or str(text).strip() == '':
        return 'neutral', 0.5
    
    try:
        text_str = str(text)[:512]  
        
        result = pipe(text_str)
        top_emotion = max(result[0], key=lambda x: x['score'])
        emotion = top_emotion['label']
        score = top_emotion['score']
        
        return emotion, round(score, 4)
                
    except Exception as e:
        print(f"Errore nell'analisi: {e}")
        return 'neutral', 0.5


def analizza_emozioni_completo(df, colonna_testo='clean_comment'):
    """
    MAIN FUNCTION: Performs a comprehensive analysis of emotions.
    
    Args:
        df: A pandas DataFrame containing the text to be analyzed
        colonna_testo: name of the column containing the texts (default: 'clean_comment')

    Returns:
        DataFrame with 2 new columns: emotion, emotion_score

    The 7 recognized emotions are:
        anger, disgust, fear, joy, neutral, sadness, surprise
    """
    
    print("="*60)
    print("EMOTIONS ANALYSIS - Modello j-hartmann")
    print("="*60)
    
    if colonna_testo not in df.columns:
        raise ValueError(f"Column '{colonna_testo}' not found! Available columns: {list(df.columns)}")

    pipe = carica_modello_emotion()

    print(f"\n✓ Model loaded successfully!")
    print(f"Starting analysis of {len(df)} texts...")
    
    emotions = []
    scores = []
    
    for idx, text in enumerate(df[colonna_testo]):
        if idx % 100 == 0:
            print(f"  Processed: {idx}/{len(df)} texts...")
        
        emotion, score = analizza_emozione(text, pipe)
        emotions.append(emotion)
        scores.append(score)

    df['emotion'] = emotions
    df['emotion_score'] = scores

    print(f"\n✓ Analysis completed: {len(df)} texts processed")

    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())
    print(f"\nGlobal average score: {df['emotion_score'].mean():.4f}")

    print("\n✅ Analysis completed!")

    return df