# ab1.py  (Replace your file content with this)
import os
import joblib
import tkinter as tk
from tkinter import messagebox, ttk, scrolledtext
import pandas as pd
import requests
from bs4 import BeautifulSoup

# ML imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

# newspaper for robust article extraction
from newspaper import Article
import nltk

MODEL_PATH = "fake_news_model.pkl"
DATA_XLSX = "fake_news_dataset.xlsx"
DATA_CSV = "fake_news_dataset.csv"

# simple suspicious keywords for explanation
SUSPICIOUS_KEYWORDS = ["miracle", "instant", "cure", "shocking", "unbelievable",
                       "conspiracy", "secret", "hidden", "exposed", "allegedly",
                       "claims", "claim", "discover", "breakthrough", "weight loss"]

# ensure punkt for newspaper
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

def read_dataset():
    """Try to load dataset from XLSX or CSV. Return (X_series, y_series) or (None, None)."""
    if os.path.exists(DATA_XLSX):
        try:
            df = pd.read_excel(DATA_XLSX)
        except Exception as e:
            print("Failed to read xlsx:", e)
            df = None
    elif os.path.exists(DATA_CSV):
        try:
            df = pd.read_csv(DATA_CSV)
        except Exception as e:
            print("Failed to read csv:", e)
            df = None
    else:
        df = None

    if df is None:
        return None, None

    # Heuristics: combine title+text if present, else use first text-like column
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'].astype(str) + " " + df['text'].astype(str)
    elif 'text' in df.columns:
        df['content'] = df['text'].astype(str)
    else:
        # fallback: choose the longest object column
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        if not obj_cols:
            df['content'] = df.iloc[:, 0].astype(str)
        else:
            # pick column with largest average length
            best = max(obj_cols, key=lambda c: df[c].astype(str).map(len).mean())
            df['content'] = df[best].astype(str)

    # label detection
    if 'label' in df.columns:
        y = df['label']
        if y.dropna().dtype.kind in 'iuf':  # numeric 0/1
            y = y.map({0: 'FAKE', 1: 'REAL'}).fillna('FAKE')
    elif 'truth' in df.columns:
        y = df['truth']
    else:
        # if no label, mark all REAL (not ideal) — fallback
        y = pd.Series(['REAL'] * len(df))

    return df['content'].fillna(""), y

def train_and_save_model():
    X, y = read_dataset()
    if X is None:
        # small fallback balanced dataset
        data = {
            'text': [
                'The economy is doing well and jobs are increasing',
                'Aliens have landed on Earth and taken over the government',
                'New vaccine proves to be 99% effective',
                'Miracle diet pill causes instant weight loss',
                'Government announces new education reforms',
                'A man claims he traveled through time and met himself',
                'NASA confirms discovery of water on Mars',
                'Scientists warn about effects of climate change',
                'Celebrity falsely reported dead on social media',
                'Conspiracy theories about moon landing exposed'
            ],
            'label': ['REAL','FAKE','REAL','FAKE','REAL','FAKE','REAL','REAL','FAKE','FAKE']
        }
        df = pd.DataFrame(data)
        X = df['text']
        y = df['label']

    # split and train
    try:
        strat = y if len(set(y)) > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=strat)
    except Exception:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.9)
    x_train_vec = vectorizer.fit_transform(x_train)

    model = PassiveAggressiveClassifier(max_iter=200)
    model.fit(x_train_vec, y_train)

    joblib.dump((vectorizer, model), MODEL_PATH)

    # try report test accuracy
    try:
        x_test_vec = vectorizer.transform(x_test)
        acc = model.score(x_test_vec, y_test)
        print(f"Model trained and saved. Test accuracy: {acc:.3f}")
    except Exception:
        print("Model trained and saved (no accuracy available).")

# train if no model found
if not os.path.exists(MODEL_PATH):
    train_and_save_model()

# load saved model
vectorizer, model = joblib.load(MODEL_PATH)

# --- Utilities for explanation and fetching ---
def explain_reason(text, url=None, prediction=None):
    text_lower = (text or "").lower()
    reasons = []
    found = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text_lower]
    if found:
        reasons.append("Contains suspicious keywords: " + ", ".join(found[:6]))

    # sensational punctuation / excessive uppercase
    if sum(1 for c in text if c.isupper()) > 60:
        reasons.append("Excessive UPPERCASE indicates sensational style.")

    if url:
        try:
            domain = url.split("//")[-1].split("/")[0].lower().split(':')[0]
            reasons.append(f"Source domain: {domain}")
        except Exception:
            pass

    if not reasons:
        if prediction == "FAKE":
            reasons.append("Model detected patterns similar to fake-news examples (sensational wording).")
        else:
            reasons.append("No obvious suspicious patterns found.")
    return " | ".join(reasons)

def fetch_article_newspaper(url):
    try:
        art = Article(url)
        art.download()
        art.parse()
        content = (art.title or "") + "\n\n" + (art.text or "")
        if content.strip():
            return content.strip()
    except Exception:
        return None

def fetch_article_requests(url):
    # fallback simple fetch: get text from <p> tags
    try:
        resp = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # prefer article tags or main
        parts = soup.find_all(['article'])
        if not parts:
            parts = soup.find_all('p')
        text = " ".join(p.get_text(separator=" ", strip=True) for p in parts[:200])
        return text if text.strip() else None
    except Exception:
        return None

def fetch_article_text(url):
    # try newspaper first, then requests+bs4 fallback
    out = fetch_article_newspaper(url)
    if out:
        return out
    return fetch_article_requests(url)

# --- GUI ---
app = tk.Tk()
app.title("📰 Fake News Detector - Upgraded")
app.geometry("820x680")
app.configure(bg="#1e1e2f")

style = ttk.Style()
style.theme_use('clam')
style.configure("TButton", foreground="white", background="#6c63ff", font=("Helvetica", 11, "bold"), padding=8)
style.map("TButton", background=[('active', '#5146d8')])

tk.Label(app, text="🧠 Fake News Detector (Text / URL)", font=("Helvetica", 20, "bold"), bg="#1e1e2f", fg="#6c63ff").pack(pady=12)

tk.Label(app, text="Paste article text here:", bg="#1e1e2f", fg="white").pack(anchor='w', padx=12)
text_input = scrolledtext.ScrolledText(app, height=10, width=95, font=("Helvetica", 11), bg="#2d2d44", fg="white", insertbackground="white", wrap="word", borderwidth=2, relief="groove")
text_input.pack(padx=12, pady=8)

tk.Label(app, text="Or paste article URL here:", bg="#1e1e2f", fg="white").pack(anchor='w', padx=12)
url_var = tk.StringVar()
url_entry = tk.Entry(app, textvariable=url_var, width=95, font=("Helvetica", 11), bg="#2d2d44", fg="white", insertbackground="white", relief="groove")
url_entry.pack(padx=12, pady=6)

result_label = tk.Label(app, text="", font=("Helvetica", 16, "bold"), bg="#1e1e2f", fg="white")
result_label.pack(pady=8)

reason_label = tk.Label(app, text="", font=("Helvetica", 11), bg="#1e1e2f", fg="white", wraplength=780, justify="left")
reason_label.pack(pady=6)

def detect_from_text(use_url_if_empty=True):
    news = text_input.get("1.0", tk.END).strip()
    url = url_var.get().strip()
    if not news and not url:
        messagebox.showwarning("Input Needed", "Please enter article text or paste an article URL.")
        return

    used_url = None
    if not news and url and use_url_if_empty:
        fetched = fetch_article_text(url)
        if not fetched:
            messagebox.showerror("Fetch Failed", "Could not fetch article text from URL.")
            return
        news = fetched
        used_url = url
        text_input.delete("1.0", tk.END)
        text_input.insert(tk.END, news)

    vec = vectorizer.transform([news])
    pred = model.predict(vec)[0]
    color = "#00e676" if pred == "REAL" else "#ff1744"
    result_label.config(text=f"🔍 News is: {pred}", fg=color)

    explanation = explain_reason(news, url=used_url or (url if url else None), prediction=pred)
    reason_label.config(text="Explanation: " + explanation)

def fetch_and_detect():
    url = url_var.get().strip()
    if not url:
        messagebox.showwarning("Input Needed", "Please paste a URL.")
        return
    fetched = fetch_article_text(url)
    if not fetched:
        messagebox.showerror("Fetch Failed", "Could not fetch article text from URL. Try another URL.")
        return
    text_input.delete("1.0", tk.END)
    text_input.insert(tk.END, fetched)
    detect_from_text(use_url_if_empty=False)

btn_frame = tk.Frame(app, bg="#1e1e2f")
btn_frame.pack(pady=10)
ttk.Button(btn_frame, text="DETECT FROM TEXT", command=detect_from_text).grid(row=0, column=0, padx=6)
ttk.Button(btn_frame, text="FETCH & DETECT FROM URL", command=fetch_and_detect).grid(row=0, column=1, padx=6)

def show_about():
    msg = ("Fake News Detector\n"
           "Model: TF-IDF + PassiveAggressiveClassifier\n"
           "Features:\n"
           "- Detect from pasted text or article URL\n"
           "- Simple explanation (keywords + source info)\n"
           "- Trained on local dataset if available (fake_news_dataset.xlsx / .csv)\n\n"
           "Note: Explanation is heuristic-based and for guidance only.")
    messagebox.showinfo("About", msg)

ttk.Button(app, text="ABOUT", command=show_about).pack(side="bottom", pady=10)

app.mainloop()
