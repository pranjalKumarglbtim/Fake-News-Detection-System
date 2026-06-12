import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, ShieldOff, Globe, FileText, AlertTriangle, CheckCircle, RefreshCw, Copy, ExternalLink } from 'lucide-react';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

export default function App() {
  const [text, setText] = useState('');
  const [url, setUrl] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('text');

  const handleSubmit = async () => {
    if (!text.trim() && !url.trim()) return;
    
    setLoading(true);
    setResult(null);
    
    try {
      const payload = { text, url, fetchUrl: activeTab === 'url' && url };
      const { data } = await axios.post(`${API_BASE}/detect`, payload, { 
        timeout: 30000,
        headers: { 'Content-Type': 'application/json' }
      });
      setResult(data);
    } catch (err) {
      setResult({ error: err.response?.data?.message || err.message || 'Detection failed' });
    } finally {
      setLoading(false);
    }
  };

const handleFetchUrl = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setResult(null);
    try {
      const { data } = await axios.post(`${API_BASE}/fetch`, { url }, { 
        timeout: 15000,
        headers: { 'Content-Type': 'application/json' }
      });
      if (data.content) {
        setText(data.content);
        // Immediately detect after fetch
        const detectResult = await axios.post(`${API_BASE}/detect`, { text: data.content, url }, { 
          timeout: 30000,
          headers: { 'Content-Type': 'application/json' }
        });
        setResult(detectResult.data);
      } else {
        setResult({ error: data.error || 'Could not fetch article' });
      }
    } catch (err) {
      setResult({ error: err.response?.data?.error || err.message || 'Fetch failed' });
    } finally {
      setLoading(false);
    }
  };

const detectWithText = async (content, usedUrl = null, skipLoading = false) => {
    if (!skipLoading) setLoading(true);
    try {
      const payload = { text: content, url: usedUrl };
      const { data } = await axios.post(`${API_BASE}/detect`, payload, { 
        timeout: 30000,
        headers: { 'Content-Type': 'application/json' }
      });
      setResult(data);
    } catch (err) {
      setResult({ error: err.response?.data?.message || err.message || 'Detection failed' });
    } finally {
      if (!skipLoading) setLoading(false);
    }
  };

  const copyToClipboard = (txt) => {
    navigator.clipboard.writeText(txt);
  };

  const resultVariants = {
    hidden: { opacity: 0, y: 30, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1, transition: { type: 'spring', stiffness: 300, damping: 25 } },
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-950 to-slate-900">
      {/* Animated background elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        {[...Array(15)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-purple-400 rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              opacity: [0, 1, 0],
              scale: [0, 1, 0],
            }}
            transition={{
              duration: 3 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={`grad${i}`}
            className="absolute w-64 h-64 rounded-full opacity-10"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: i % 2 === 0 ? 'linear-gradient(135deg, #7c3aed, #ec4899)' : 'linear-gradient(135deg, #06b6d4, #7c3aed)',
            }}
            animate={{
              x: [0, 50, -50, 0],
              y: [0, 50, -50, 0],
            }}
            transition={{
              duration: 15 + i * 5,
              repeat: Infinity,
            }}
          />
        ))}
      </div>

      <div className="relative z-10 max-w-4xl mx-auto px-4 py-10">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-10"
        >
          <motion.div className="inline-flex items-center gap-3 mb-4">
            <motion.div
              animate={{ rotate: [0, 5, -5, 0] }}
              transition={{ duration: 4, repeat: Infinity }}
            >
              <Shield className="w-12 h-12 text-purple-400" />
            </motion.div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-300 to-pink-300 bg-clip-text text-transparent">
              Fake News Detector
            </h1>
          </motion.div>
          <p className="text-slate-400 max-w-2xl mx-auto">
            AI-powered tool to detect misinformation in news articles. Paste text or a URL to analyze credibility.
          </p>
        </motion.div>

        {/* Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="flex gap-2 mb-6 p-2 bg-slate-800/50 rounded-xl backdrop-blur"
        >
          <motion.button
            onClick={() => setActiveTab('text')}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'text'
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-600/30'
                : 'bg-transparent text-slate-400 hover:text-white'
            }`}
            whileTap={{ scale: 0.98 }}
          >
            <FileText className="w-5 h-5" />
            Paste Text
          </motion.button>
          <motion.button
            onClick={() => setActiveTab('url')}
            className={`flex-1 flex items-center justify-center gap-2 px-4 py-3 rounded-lg font-medium transition-all ${
              activeTab === 'url'
                ? 'bg-purple-600 text-white shadow-lg shadow-purple-600/30'
                : 'bg-transparent text-slate-400 hover:text-white'
            }`}
            whileTap={{ scale: 0.98 }}
          >
            <Globe className="w-5 h-5" />
            Fetch URL
          </motion.button>
        </motion.div>

        {/* Input Area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="space-y-4"
        >
          <AnimatePresence>
            {activeTab === 'text' && (
              <motion.div
                key="text-input"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
              >
                <textarea
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste news article text here..."
                  className="w-full h-52 px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:border-purple-500 resize-none"
                />
              </motion.div>
            )}
          </AnimatePresence>

          <AnimatePresence>
            {activeTab === 'url' && (
              <motion.div
                key="url-input"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.3 }}
                className="space-y-3"
              >
                <div className="flex gap-2">
                  <input
                    type="url"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="https://example.com/article..."
                    className="flex-1 px-4 py-3 bg-slate-800/50 border border-slate-700 rounded-xl text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
                  />
                  <motion.button
                    onClick={handleFetchUrl}
                    disabled={loading || !url.trim()}
                    className="px-4 py-3 bg-purple-600 rounded-xl text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    whileTap={{ scale: 0.98 }}
                  >
                    {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <ExternalLink className="w-4 h-4" />}
                  </motion.button>
                </div>
                {url && !loading && (
                  <motion.p
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="text-xs text-slate-500"
                  >
                    Click "Detect" or press Enter to analyze
                  </motion.p>
                )}
              </motion.div>
            )}
          </AnimatePresence>

          <motion.button
            onClick={handleSubmit}
            disabled={loading || (!text.trim() && !url.trim())}
            className="w-full py-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-xl text-white font-bold text-lg shadow-lg shadow-purple-600/30 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-3"
            whileTap={{ scale: 0.98 }}
          >
            {loading ? (
              <>
                <RefreshCw className="w-5 h-5 animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                🔍 Detect Fake News
              </>
            )}
          </motion.button>
        </motion.div>

        {/* Result */}
        <AnimatePresence>
          {result && (
            <motion.div
              key="result"
              variants={resultVariants}
              initial="hidden"
              animate="visible"
              exit="hidden"
              className="mt-8"
            >
              {result.error ? (
                <motion.div className="bg-red-500/10 border border-red-500/30 rounded-2xl p-6">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <div className="text-red-300">{result.error}</div>
                  </div>
                </motion.div>
              ) : result.prediction ? (
                <motion.div className="space-y-4">
                  {/* Prediction */}
                  <motion.div
                    className={`rounded-2xl p-6 border ${
                      result.prediction === 'REAL'
                        ? 'bg-emerald-500/10 border-emerald-500/30'
                        : 'bg-red-500/10 border-red-500/30'
                    }`}
                    initial={{ scale: 0.95 }}
                    animate={{ scale: 1 }}
                    transition={{ type: 'spring', stiffness: 400, damping: 25 }}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        {result.prediction === 'REAL' ? (
                          <CheckCircle className="w-8 h-8 text-emerald-400" />
                        ) : (
                          <ShieldOff className="w-8 h-8 text-red-400" />
                        )}
                        <div>
                          <div className="text-slate-400 text-sm">Prediction</div>
                          <div className={`text-3xl font-bold ${
                            result.prediction === 'REAL' ? 'text-emerald-400' : 'text-red-400'
                          }`}>
                            {result.prediction}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-slate-400 text-sm">Confidence</div>
                        <div className="text-2xl font-bold text-white">{result.confidence}%</div>
                      </div>
                    </div>

                    {/* Dual Percentage Bars */}
                    <div className="mt-6 space-y-3">
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-emerald-400 font-medium">REAL</span>
                          <span className="text-slate-300">{result.real_percentage}%</span>
                        </div>
                        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                          <motion.div
                            className="h-full bg-emerald-500"
                            initial={{ width: 0 }}
                            animate={{ width: `${result.real_percentage}%` }}
                            transition={{ duration: 0.8, ease: 'easeOut' }}
                          />
                        </div>
                      </div>
                      <div>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-red-400 font-medium">FAKE</span>
                          <span className="text-slate-300">{result.fake_percentage}%</span>
                        </div>
                        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                          <motion.div
                            className="h-full bg-red-500"
                            initial={{ width: 0 }}
                            animate={{ width: `${result.fake_percentage}%` }}
                            transition={{ duration: 0.8, ease: 'easeOut' }}
                          />
                        </div>
                      </div>
                    </div>
                  </motion.div>

                  {/* Reasons */}
                  <motion.div
                    className="bg-slate-800/50 border border-slate-700 rounded-2xl p-6"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.4, delay: 0.1 }}
                  >
                    <div className="text-slate-400 text-sm mb-3">Analysis Reasons:</div>
                    <ul className="space-y-2">
                      {result.reasons?.map((reason, i) => (
                        <motion.li
                          key={i}
                          className="flex items-start gap-2 text-slate-200"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ duration: 0.3, delay: i * 0.1 }}
                        >
                          <span className="text-purple-400 mt-0.5">•</span>
                          {reason}
                        </motion.li>
                      ))}
                    </ul>
                  </motion.div>

                  {/* Linguistic Analysis */}
                  {result.linguistic_analysis && (
                    <motion.div
                      className="bg-slate-800/50 border border-slate-600 rounded-2xl p-6"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.4, delay: 0.2 }}
                    >
                      <div className="text-slate-400 text-sm mb-4">Deep Linguistic Analysis</div>
                      <div className="grid grid-cols-2 gap-4">
                        {[
                          { name: "Semantic Meaning", key: "semantic_meaning", desc: "Factual density & coherence" },
                          { name: "Structure", key: "structure", desc: "Journalistic integrity" },
                          { name: "Alignment", key: "alignment", desc: "Source credibility" },
                          { name: "Emotion", key: "emotion", desc: "Sensationalism level" }
                        ].map((item, i) => {
                          const val = result.linguistic_analysis[item.key]
                          const isGood = item.key !== "emotion" ? val >= 50 : val <= 50
                          return (
                            <motion.div
                              key={item.key}
                              className="bg-slate-900/50 rounded-lg p-3"
                              initial={{ opacity: 0, scale: 0.9 }}
                              animate={{ opacity: 1, scale: 1 }}
                              transition={{ duration: 0.3, delay: i * 0.1 }}
                            >
                              <div className="text-xs text-slate-400 mb-1">{item.name}</div>
                              <div className="flex items-center gap-2">
                                <div className="text-lg font-bold text-white">{val}</div>
                                <div className="text-xs text-slate-500">({item.desc})</div>
                              </div>
                              <div className="h-1.5 bg-slate-700 rounded-full mt-2">
                                <div
                                  className={`h-full rounded-full ${isGood ? 'bg-emerald-500' : 'bg-red-500'}`}
                                  style={{ width: `${val}%` }}
                                />
                              </div>
                            </motion.div>
                          )
                        })}
                      </div>
                    </motion.div>
                  )}

                  {/* Stats */}
                  <motion.div
                    className="grid grid-cols-2 gap-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.4, delay: 0.2 }}
                  >
                    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                      <div className="text-2xl font-bold text-white">{result.word_count}</div>
                      <div className="text-slate-400 text-sm">Words Analyzed</div>
                    </div>
                    <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-4 text-center">
                      <div className="text-2xl font-bold text-white">TF-IDF</div>
                      <div className="text-slate-400 text-sm">ML Model</div>
                    </div>
                  </motion.div>

                  {/* Text Preview */}
                  <motion.div
                    className="bg-slate-900/50 border border-slate-700 rounded-xl p-4"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ duration: 0.4, delay: 0.3 }}
                  >
                    <div className="text-slate-400 text-sm mb-2">Text Preview:</div>
                    <div className="text-slate-300 text-sm leading-relaxed line-clamp-4">
                      {result.text_preview}
                    </div>
                    <button
                      onClick={() => copyToClipboard(result.text_preview)}
                      className="mt-2 text-xs text-purple-400 hover:text-purple-300 flex items-center gap-1"
                    >
                      <Copy className="w-3 h-3" />
                      Copy to clipboard
                    </button>
</motion.div>
                  </motion.div>
                ) : null}
              </motion.div>
            )}
          </AnimatePresence>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="mt-12 text-center text-slate-500 text-xs"
        >
          <p>Powered by TF-IDF + PassiveAggressiveClassifier • NLP-based detection</p>
        </motion.div>
      </div>
    </div>
  );
}