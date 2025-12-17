# backend/app.py
"""
Voice Assistant backend (Flask).
Behavior:
 - If user query clearly matches a stored question (exact phrase, token-match, or fuzzy match),
   return that stored CSV answer immediately.
 - Otherwise fall back to combined word+char TF-IDF with lexical overlap boosting.
CSV expected at backend/questions.csv (same folder as this file).
"""

import os
import pickle
import re
from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack
from difflib import SequenceMatcher

# --------------------------
# CONFIG
# --------------------------
CSV_PATH = os.path.join(os.path.dirname(__file__), "questions.csv")
VECTORS_CACHE = os.path.join(os.path.dirname(__file__), "vector_cache.pkl")
FRONTEND_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
THRESHOLD = float(os.environ.get("QA_SIM_THRESHOLD", "0.40"))   # TF-IDF fallback threshold
FORCE_REBUILD = os.environ.get("FORCE_REBUILD", "0").lower() in ("1", "true", "yes")

app = Flask(__name__, static_folder=FRONTEND_FOLDER, static_url_path='')

# Globals populated at startup
vectorizers = None
q_vectors = None
questions = []
answers = []

_token_re = re.compile(r"\w+")

def normalize_text_tokens(s: str):
    return _token_re.findall(s.lower())

def clean_text(s: str):
    return " ".join(normalize_text_tokens(s))

def _find_q_a_columns(df: pd.DataFrame):
    cols_lower = [c.lower().strip() for c in df.columns]
    qcol = None; acol = None
    for c_lower, orig in zip(cols_lower, df.columns):
        if c_lower in ("question", "ques", "q", "queries", "query", "questions"):
            qcol = orig
        if c_lower in ("answer", "ans", "a", "response", "responses", "reply"):
            acol = orig
    return qcol, acol

def load_qa_and_build_vectors(csv_path=CSV_PATH, cache_path=VECTORS_CACHE, force_rebuild=False):
    global vectorizers, q_vectors, questions, answers

    if force_rebuild and os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            app.logger.info(f"Removed cache because FORCE_REBUILD=True: {cache_path}")
        except Exception:
            pass

    # Try load cache
    if not force_rebuild and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            vectorizers = cache["vectorizers"]
            q_vectors = cache["q_vectors"]
            questions = cache["questions"]
            answers = cache["answers"]
            app.logger.info(f"Loaded TF-IDF cache from {cache_path} (questions: {len(questions)})")
            return
        except Exception:
            app.logger.warning("Failed to load vector cache, will rebuild.")

    # Read CSV
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"QA CSV not found at {csv_path}. Place the file there or update CSV_PATH.")
    df = pd.read_csv(csv_path)

    qcol, acol = _find_q_a_columns(df)
    if qcol is None or acol is None:
        raise RuntimeError("questions.csv must have question and answer columns. Found: " + ", ".join(df.columns))

    # Clean and dedupe
    df = df.drop_duplicates(subset=[qcol])
    df[qcol] = df[qcol].fillna("").astype(str)
    df = df[df[qcol].str.strip() != ""]
    questions = df[qcol].astype(str).tolist()
    answers = df[acol].fillna("").astype(str).tolist()

    if len(questions) == 0:
        raise RuntimeError("No valid questions found after cleaning CSV. Check your questions.csv file.")

    # Build TF-IDF: word (1-2) + char (3-5)
    word_vec = TfidfVectorizer(stop_words='english', analyzer='word', ngram_range=(1,2), sublinear_tf=True)
    W = word_vec.fit_transform(questions)
    char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), sublinear_tf=True)
    C = char_vec.fit_transform(questions)
    q_vectors = hstack([W, C]).tocsr()
    vectorizers = {"word": word_vec, "char": char_vec}

    try:
        with open(cache_path, "wb") as f:
            pickle.dump({"vectorizers": vectorizers, "q_vectors": q_vectors, "questions": questions, "answers": answers}, f)
        app.logger.info(f"Saved TF-IDF cache to {cache_path}")
    except Exception:
        app.logger.warning("Failed to save TF-IDF cache (non-fatal).")

# --------------------------
# Matching logic
# --------------------------
def _fuzzy_ratio(a: str, b: str):
    return SequenceMatcher(None, a, b).ratio()

def find_best_answer(user_text: str, top_k=6):
    """
    1) Clean + exact phrase (if user_text is substring of any stored question) -> return that answer
    2) Token-level strong match: if all important tokens from user appear in a question -> return that answer
    3) Fuzzy ratio: if SequenceMatcher ratio >= 0.82 with any stored question -> return that answer
    4) TF-IDF fallback with lexical overlap boosting
    """
    global vectorizers, q_vectors, questions, answers, THRESHOLD

    if vectorizers is None or q_vectors is None:
        raise RuntimeError("Vectors not initialized; ensure load_qa_and_build_vectors ran successfully.")

    if not user_text or not str(user_text).strip():
        return {"answer": "I didn't catch that. Please say your question again.", "score": 0.0, "match_idx": None, "debug": []}

    u_clean = clean_text(user_text)            # normalized user text tokens joined
    u_tokens = set(u_clean.split())

    # 1) exact phrase (lowercase substring) — highest priority
    for i, q in enumerate(questions):
        if u_clean and u_clean in clean_text(q):
            app.logger.info(f"Exact-substring match user->{i}")
            return {"answer": answers[i], "score": 1.0, "match_idx": i, "matched_question": questions[i], "debug":[{"idx":i,"reason":"exact_substring"}]}

    # 2) token-level strong match: if >50% of user tokens appear in stored question tokens, prefer it
    if len(u_tokens) > 0:
        for i, q in enumerate(questions):
            q_tokens = set(normalize_text_tokens(q))
            if len(u_tokens) == 0:
                continue
            overlap = len(u_tokens.intersection(q_tokens)) / float(len(u_tokens))
            if overlap >= 0.6:   # 60% of tokens match -> strong match
                app.logger.info(f"Token-overlap strong match user->{i} overlap={overlap:.2f}")
                return {"answer": answers[i], "score": 0.98, "match_idx": i, "matched_question": questions[i], "debug":[{"idx":i,"reason":"token_overlap","overlap":overlap}]}

    # 3) fuzzy ratio match: if similarity between user string and stored question is high
    #    (use cleaned forms for ratio)
    for i, q in enumerate(questions):
        ratio = _fuzzy_ratio(u_clean, clean_text(q))
        if ratio >= 0.82:   # tuneable threshold (0.82 is strict)
            app.logger.info(f"Fuzzy match user->{i} ratio={ratio:.3f}")
            return {"answer": answers[i], "score": ratio, "match_idx": i, "matched_question": questions[i], "debug":[{"idx":i,"reason":"fuzzy_ratio","ratio":ratio}]}

    # 4) TF-IDF fallback with lexical boost
    uw = vectorizers["word"].transform([user_text])
    uc = vectorizers["char"].transform([user_text])
    user_vec = hstack([uw, uc]).tocsr()
    sims = cosine_similarity(user_vec, q_vectors).flatten()

    # choose top_k by base score then boost by token overlap
    top_idxs = sims.argsort()[::-1][:top_k]
    debug = []
    boosted = []

    exact_token_bonus = 0.18
    partial_overlap_bonus = 0.08

    for i in top_idxs:
        base = float(sims[i])
        q_tokens = set(normalize_text_tokens(questions[int(i)]))
        overlap_count = len(u_tokens.intersection(q_tokens))
        overlap_ratio = (overlap_count / float(len(u_tokens))) if len(u_tokens) > 0 else 0.0
        bonus = 0.0
        if overlap_count > 0:
            bonus += exact_token_bonus
            bonus += partial_overlap_bonus * overlap_ratio
        new_score = base + bonus
        boosted.append((int(i), new_score, base, overlap_count, overlap_ratio))
        debug.append({"idx": int(i), "question": questions[int(i)], "answer": answers[int(i)], "base_score": base, "overlap_count": overlap_count, "overlap_ratio": overlap_ratio, "boosted_score": new_score})

    boosted.sort(key=lambda x: x[1], reverse=True)
    best_idx, best_score, base_score, oc, oratio = boosted[0]

    app.logger.info(f"TF-IDF chosen idx {best_idx} base={base_score:.3f} boosted={best_score:.3f} overlap={oc}")

    if best_score < THRESHOLD:
        return {"answer": "Sorry, I don't have a confident answer. Try rephrasing your question.", "score": best_score, "match_idx": None, "debug": debug}

    return {"answer": answers[int(best_idx)], "score": best_score, "match_idx": int(best_idx), "matched_question": questions[int(best_idx)], "debug": debug}

# --------------------------
# Startup: load vectors
# --------------------------
try:
    load_qa_and_build_vectors(force_rebuild=FORCE_REBUILD)
except Exception as e:
    app.logger.exception("Failed to load or build QA vectors: " + str(e))
    raise

# --------------------------
# Flask endpoints
# --------------------------
@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.get_json(force=True)
    text = data.get('text', '')
    try:
        res = find_best_answer(text, top_k=8)
    except Exception as e:
        app.logger.exception("Error while finding best answer: " + str(e))
        return jsonify({"answer": "Internal error", "score": 0.0, "match_idx": None, "debug": []}), 500
    return jsonify(res)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_frontend(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
