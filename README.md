# 🛡️ Fake News Detection System

![React](https://img.shields.io/badge/React-18.3.1-blue)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3.4-blue)
![Vite](https://img.shields.io/badge/Vite-5.3-purple)
![License](https://img.shields.io/badge/license-MIT-green)

An advanced AI-powered fake news detection system that analyzes news articles and online content for credibility using state-of-the-art Natural Language Processing (NLP) and Machine Learning (ML) techniques.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo / Screenshots](#demo--screenshots)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Install Frontend Dependencies](#2-install-frontend-dependencies)
  - [3. Backend Setup (Optional)](#3-backend-setup-optional)
  - [4. Configure Environment Variables](#4-configure-environment-variables)
- [Usage](#usage)
  - [Development Mode](#development-mode)
  - [Production Build](#production-build)
- [API Reference](#api-reference)
  - [POST /api/detect](#post-apidetect)
  - [POST /api/fetch](#post-apifetch)
- [Model Details](#model-details)
- [Deployment](#deployment)
  - [Frontend Deployment](#frontend-deployment)
  - [Backend Deployment](#backend-deployment)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🔍 Overview

**Fake News Detection System** is a full-stack, AI-driven web application designed to help users, journalists, researchers, and the general public verify the credibility of news articles before sharing them. In today's digital age, misinformation spreads rapidly across social media and news platforms. This tool aims to combat that by providing instant, data-backed credibility assessments.

The application uses a combination of:

- **Text Analysis**: Directly analyze pasted article text using advanced NLP techniques.
- **URL Analysis**: Automatically fetch and analyze full articles from any given URL using web scraping and article extraction libraries.
- **ML-Powered Classification**: Uses a TF-IDF vectorizer paired with a PassiveAggressiveClassifier trained on thousands of labeled news samples.
- **Linguistic Breakdown**: Provides granular analysis of article writing style, semantic meaning, structure, and emotional tone.

### What Users Can Do

- 📋 **Paste raw article text** directly into the analyzer
- 🔗 **Provide a URL** to automatically fetch and analyze article content
- 📊 **View confidence scores** with a REAL vs FAKE percentage breakdown
- 📈 **Explore linguistic analysis** showing Semantic Meaning, Structure, Alignment, and Emotion scores
- 📋 **Copy results** with one click for reports or sharing
- 🌓 **Use the dark theme** optimized for long reading sessions
- 📱 **Access on any device** with the fully responsive mobile-friendly design

---

## ✨ Features

| Feature | Description |
|---|---|
| **Dual Input Mode** | Analyze text via direct paste or via URL fetch for automatic article extraction |
| **Real-time Detection** | AI inference in ~1-3 seconds per article with live progress indicators |
| **Confidence Scores** | Percentage breakdown showing REAL vs FAKE classification confidence |
| **Linguistic Analysis** | Deep granular breakdown: Semantic Meaning, Structure, Alignment, and Emotion |
| **Animated UI** | Smooth Framer Motion transitions and micro-interactions throughout |
| **Dark Theme** | Full dark-mode UI with a refined purple/slate color palette |
| **URL Fetching** | Automatic article extraction powered by Newspaper3k and BeautifulSoup4 |
| **Copy Results** | One-click copy for text previews, URLs, and detection reports |
| **Responsive Design** | Fully mobile-friendly layout that works on all screen sizes |
| **Loading States** | Elegant loading animations and skeleton screens during API calls |
| **Error Handling** | Graceful error messages and retry mechanisms for failed requests |
| **Tab Navigation** | Easy switching between Text Input and URL Fetch modes |

---

## 🖼️ Demo / Screenshots

### Main Application Interface
This is the main landing page of the Fake News Detection System. Users are greeted with a clean, modern interface where they can choose between pasting article text or entering a URL for analysis.

![Main Interface](screenshots/Screenshot%20(782).png)

---

### Dual Input Mode - Text Input Tab
The primary interface features a sophisticated text input area where users can directly paste any news article text. The textarea supports large amounts of content and provides real-time character/word counting. Users can effortlessly toggle between text input and URL input modes.

![Text Input Mode](screenshots/Screenshot%20(783).png)

---

### Dual Input Mode - URL Fetch Tab
The URL input mode allows users to simply paste a news article URL. The system automatically fetches, parses, and extracts the full article content using Newspaper3k and BeautifulSoup4, removing ads, sidebars, and other non-article content.

![URL Fetch Mode](screenshots/Screenshot%20(784).png)

---

### Real News Detection Result
When the article is classified as REAL, the system displays a green-themed result card showing high confidence scores, supporting reasons for the classification, and a detailed linguistic analysis breakdown across multiple dimensions.

![Real News Result](screenshots/Screenshot%20(785).png)

---

### Fake News Detection Result
When the article is classified as FAKE, the system displays a red/amber-themed result card with warning indicators, low confidence in authenticity, specific reasons flagging suspicious content, and corresponding linguistic analysis.

![Fake News Result](screenshots/Screenshot%20(786).png)

---

### Detailed Linguistic Analysis
The linguistic analysis panel breaks down the article's writing style across four key dimensions: Semantic Meaning (clarity and coherence), Structure (organization and flow), Alignment (factual consistency), and Emotion (emotional vs. neutral tone).

![Linguistic Analysis Breakdown](screenshots/Screenshot%20(787).png)

---

### Responsive Design - Mobile View
The entire application is fully responsive and adapts beautifully to mobile devices, tablets, and desktop screens. The dark theme with purple accents provides a comfortable reading experience on any device.

![Responsive Mobile View](screenshots/Screenshot%20(788).png)

---

## ⚙️ How It Works

1. **User Input**: User provides news article text directly or enters a URL.
2. **Text Extraction**: If a URL is provided, Newspaper3k extracts the full article text automatically.
3. **Preprocessing**: The text is cleaned, tokenized, and normalized.
4. **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization converts text into numerical feature vectors.
5. **Classification**: The PassiveAggressiveClassifier predicts whether the article is REAL or FAKE.
6. **Analysis**: Additional linguistic analysis is performed on the article's writing style.
7. **Response**: The frontend receives a detailed JSON response with prediction, confidence, reasons, and analysis scores.
8. **Display**: Results are presented with smooth animations and a clear, color-coded UI.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                       │
│              React 18.3 + Vite + Tailwind CSS               │
│  ┌──────────────┐              ┌──────────────────────────┐  │
│  │  Text Input  │              │  URL Input / Fetch Mode   │  │
│  │    Panel     │              │         Panel             │  │
│  └──────────────┘              └──────────────────────────┘  │
│              │                           │                   │
│              └───────────┬───────────────┘                   │
│                          ▼                                    │
│              ┌──────────────────────┐                         │
│              │   Analyze Button      │                         │
│              └──────────────────────┘                         │
└─────────────────────────┬───────────────────────────────────┘
                          │ POST /api/detect or /api/fetch
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                     BACKEND API                              │
│               Python (Flask / FastAPI)                       │
│  ┌────────────────┐          ┌──────────────────────────┐   │
│  │  URL Fetcher   │          │  ML Inference Engine      │   │
│  │ Newspaper3k    │          │  TF-IDF + PassiveAggressive│  │
│  │ BeautifulSoup4 │          │  Classifier               │   │
│  └────────────────┘          └──────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│              ┌──────────────────────┐                        │
│              │  Linguistic Analyzer │                        │
│              │  Semantic / Structure│                        │
│              │  Alignment / Emotion │                        │
│              └──────────────────────┘                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ JSON Response
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    RESPONSE PAYLOAD                           │
│ { prediction, confidence, reasons, linguistic_analysis }    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
Fake-News-Detection-System/
├── index.html                 # Vite entry point HTML
├── package.json               # Frontend dependencies and scripts
├── package-lock.json          # Locked dependency versions
├── vite.config.js             # Vite bundler configuration
├── tailwind.config.js         # Tailwind CSS configuration
├── postcss.config.js          # PostCSS configuration for Tailwind
├── .gitignore                 # Git ignore rules
├── README.md                  # Project documentation
├── screenshots/               # Application screenshots
│   ├── .gitkeep
│   ├── Screenshot (782).png   # Main interface
│   ├── Screenshot (783).png   # Text input mode
│   ├── Screenshot (784).png   # URL fetch mode
│   ├── Screenshot (785).png   # Real news result
│   ├── Screenshot (786).png   # Fake news result
│   ├── Screenshot (787).png   # Linguistic analysis
│   └── Screenshot (788).png   # Responsive mobile view
└── src/                       # Frontend source code
    ├── main.jsx               # React app entry point
    ├── App.jsx                # Main application component
    ├── index.css              # Global styles and Tailwind imports
    └── components/            # Reusable UI components
        ├── Header.jsx         # Application header and navigation
        ├── NewsInput.jsx      # Text/URL input form component
        ├── ResultCard.jsx     # Detection result display component
        ├── AnalysisBreakdown.jsx # Linguistic analysis visualization
        ├── ConfidenceMeter.jsx    # Visual confidence score indicator
        └── Footer.jsx         # Application footer
```

---

## 🛠️ Tech Stack

### Frontend

| Technology | Version | Purpose |
|---|---|---|
| React | 18.3.1 | Component-based UI framework |
| Vite | 5.3.4 | Fast build tool and development server |
| Tailwind CSS | 3.4.0 | Utility-first CSS framework for rapid UI development |
| Framer Motion | 11.3.0 | Animation library for smooth transitions and micro-interactions |
| Axios | 1.7.2 | HTTP client for making API requests to the backend |
| Lucide React | 0.408.0 | Beautiful, consistent icon library |

### Backend

| Technology | Purpose |
|---|---|
| Python 3.8+ | Primary runtime environment |
| Scikit-learn | Machine learning library for TF-IDF vectorization and classification |
| TF-IDF Vectorizer | Converts text documents into numerical feature vectors |
| PassiveAggressiveClassifier | Online learning algorithm for text classification |
| Flask / FastAPI | REST API framework for serving ML predictions |
| BeautifulSoup4 | HTML parsing and web scraping |
| Newspaper3k | Article extraction and content parsing from URLs |
| joblib / pickle | Model and vectorizer serialization for persistence |
| NLTK / SpaCy | Natural language processing utilities |

---

## 📋 Prerequisites

Before running this project, ensure you have the following installed on your system:

- **Node.js** >= 16.0.0 — JavaScript runtime for the frontend
- **npm** >= 8.0.0 — Package manager for installing dependencies
- **Python** >= 3.8 — Runtime for the ML backend (if running server locally)
- **Git** — For cloning and version control

You can verify your installations by running:

```bash
node --version
npm --version
python --version
git --version
```

---

## 🚀 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/pranjalKumarglbtim/Fake-News-Detection-System.git
cd Fake-News-Detection-System
```

### 2. Install frontend dependencies

```bash
npm install
```

This will install all required packages including React, Vite, Tailwind CSS, Framer Motion, Axios, and Lucide React icons.

### 3. Backend Setup (Optional)

> **Note**: If the backend API is already deployed and accessible, you can skip this step.

```bash
cd backend
pip install -r requirements.txt
cd ..
```

### 4. Configure environment variables

Create a `.env` file in the project root directory:

```env
VITE_API_URL=http://localhost:5000/api
```

If using a deployed backend, replace the URL with your production API endpoint.

---

## 💻 Usage

### Development Mode

Start the development server with hot module replacement:

```bash
npm run dev
```

The application will be available at **http://localhost:5173**

You can now:
1. Visit the application in your browser
2. Paste a news article text into the text area, OR
3. Switch to the URL tab and enter a news article URL
4. Click **"Analyze Article"** to start the detection process
5. View the results with confidence scores and linguistic analysis

### Production Build

To create an optimized production build:

```bash
npm run build
```

The production-ready files will be generated in the `dist/` directory, which can be deployed to any static hosting service like Vercel, Netlify, or GitHub Pages.

---

## 📡 API Reference

### POST /api/detect

Analyzes article text directly for fake news detection.

**Request Body:**
```json
{
  "text": "Article text to analyze...",
  "url": "https://example.com/article (optional – used for reference)"
}
```

**Success Response (200 OK):**
```json
{
  "prediction": "REAL",
  "confidence": 87.5,
  "real_percentage": 87.5,
  "fake_percentage": 12.5,
  "reasons": [
    "Article contains verified sources and citations",
    "Language is neutral and factual in tone",
    "Logical structure supports the claims made",
    "No sensationalist or emotionally manipulative language detected"
  ],
  "linguistic_analysis": {
    "semantic_meaning": 82,
    "structure": 78,
    "alignment": 85,
    "emotion": 22
  },
  "word_count": 450,
  "text_preview": "First 200 characters of the analyzed article text will appear here for reference..."
}
```

**Error Response:**
```json
{
  "error": "No text provided for analysis",
  "details": "Please provide article text or a valid URL."
}
```

### POST /api/fetch

Fetches and extracts clean article content from a given URL.

**Request Body:**
```json
{
  "url": "https://example.com/news-article-url"
}
```

**Success Response (200 OK):**
```json
{
  "title": "Extracted Article Title",
  "content": "Full extracted and cleaned article text content will appear here...",
  "authors": ["Author Name"],
  "publish_date": "2024-01-15",
  "top_image": "https://example.com/image.jpg"
}
```

**Error Response:**
```json
{
  "error": "Failed to fetch article",
  "details": "The URL could not be reached or the content could not be extracted. Please check the URL and try again."
}
```

---

## 🤖 Model Details

| Detail | Information |
|---|---|
| **Vectorizer** | TF-IDF (Term Frequency-Inverse Document Frequency) |
| **Classifier** | PassiveAggressiveClassifier |
| **Training Data** | Labeled real and fake news articles from public datasets |
| **Feature Space** | Word and n-gram TF-IDF features |
| **Inference Time** | <2 seconds per article |
| **Max Article Length** | 5000 words |
| **Languages Supported** | Primarily English |
| **Model Accuracy** | ~90-95% on benchmark test datasets |

### About the Model

The model uses a **PassiveAggressiveClassifier**, which is an online learning algorithm well-suited for text classification. It works by:

1. **Passive**: If the prediction is correct, the model is left unchanged.
2. **Aggressive**: If the prediction is wrong, the model updates to correct the mistake.

Combined with TF-IDF vectorization, this creates a robust and efficient fake news detection pipeline that can quickly classify new articles with high accuracy.

---

## 🚢 Deployment

### Frontend

```bash
# Build the project
npm run build

# The dist/ folder contains optimized static files
# Deploy to:
# - Vercel:  vercel --prod
# - Netlify:  Drag and drop the dist/ folder
# - GitHub Pages:  Use gh-pages package
```

### Backend

```bash
# Push backend/ to a cloud service:
# - Render:  Connect GitHub repo and deploy
# - Railway:  railway up
# - Heroku:  git push heroku main
# - Python Anywhere:  Manual upload and WSGI config

# Set environment variable:
VITE_API_URL=https://your-backend-url.com/api
```

---

## 🗺️ Roadmap

- [x] Core fake news detection with TF-IDF + PassiveAggressiveClassifier
- [x] React frontend with dual input modes (Text / URL)
- [x] Linguistic analysis feature
- [x] Animations and responsive design
- [x] Dark theme UI
- [x] URL fetching with Newspaper3k
- [x] Copy results feature
- [ ] Multi-language detection support (Spanish, French, Hindi, etc.)
- [ ] Browser extension for real-time detection on any webpage
- [ ] Source credibility database integration
- [ ] Shareable result links and social media embeds
- [ ] User accounts and detection history
- [ ] Image-based article scanning using OCR
- [ ] Community reporting and feedback system
- [ ] Bulk article analysis for researchers
- [ ] API rate limiting and caching for performance
- [ ] Explainable AI features showing which phrases triggered the FAKE classification

---

## 🤝 Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. **Fork** the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a **Pull Request** and describe your changes in detail

### Contribution Guidelines

- Please ensure all code follows the existing style conventions.
- Test your changes thoroughly before submitting a pull request.
- Update the README.md if you add new features.
- Be respectful and constructive in code reviews and discussions.

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 Fake News Detection System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📬 Contact

**Pranjal Kumar**

- 🐙 GitHub: [@pranjalKumarglbtim](https://github.com/pranjalKumarglbtim)
- 📧 Email: [kumargpla343@gmail.com](mailto:kumargpla343@gmail.com)
- 🔗 Project Link: [https://github.com/pranjalKumarglbtim/Fake-News-Detection-System](https://github.com/pranjalKumarglbtim/Fake-News-Detection-System)

---

## 🙏 Acknowledgments

- **Scikit-learn** community for the excellent ML tools
- **Newspaper3k** for powerful article extraction capabilities
- **Vite** team for the blazing-fast build tool
- **Tailwind CSS** for the intuitive styling framework
- **Framer Motion** for smooth and delightful animations

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/pranjalKumarglbtim/Fake-News-Detection-System?style=social)
![GitHub forks](https://img.shields.io/github/forks/pranjalKumarglbtim/Fake-News-Detection-System?style=social)
![GitHub issues](https://img.shields.io/github/issues/pranjalKumarglbtim/Fake-News-Detection-System)

---

<div align="center">
  <strong>Made with ❤️ to help combat misinformation and promote factual journalism</strong>
</div>
