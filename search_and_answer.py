"""
案件3 RAGパイプライン - 検索+回答生成スクリプト
質問 -> Embedding -> コサイン類似度で関連チャンクを検索 -> Claude APIで出典付き回答を生成
"""
import json
import os
import pickle
import sys

import numpy as np
from anthropic import Anthropic

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = "claude-sonnet-4-5-20250929"

client = Anthropic(api_key=ANTHROPIC_API_KEY)

with open("chunks_backup.json", "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)
with open("vectorizer.pkl", "rb") as f:
    VECTORIZER = pickle.load(f)

CHUNK_EMBEDDINGS = np.array([c["embedding"] for c in CHUNKS])
CHUNK_NORMS = np.linalg.norm(CHUNK_EMBEDDINGS, axis=1, keepdims=True)
CHUNK_NORMS[CHUNK_NORMS == 0] = 1e-9


def embed_query(text):
    # VECTORIZERはTfidfVectorizer→TruncatedSVDのパイプライン。
    # TruncatedSVDの出力は(疎行列ではなく)もともと密なndarrayなので、
    # 単体のTfidfVectorizerだった頃と違い.toarray()は不要(呼ぶとエラーになる)。
    vec = VECTORIZER.transform([text])[0]
    if len(vec) < 384:
        vec = np.concatenate([vec, np.zeros(384 - len(vec))])
    return vec


def search(query, top_k=8):
    """pgvectorのcosine similarity検索を模したローカル実装。"""
    q = embed_query(query)
    q_norm = np.linalg.norm(q) or 1e-9
    sims = (CHUNK_EMBEDDINGS @ q) / (CHUNK_NORMS.flatten() * q_norm)
    top_idx = np.argsort(-sims)[:top_k]
    return [(CHUNKS[i], float(sims[i])) for i in top_idx]


SYSTEM_PROMPT = """あなたは株式会社テックブリッジの社内文書検索AIです。
以下のルールを厳守してください:
1. 提供された「参考文書」に書かれている内容のみを根拠に回答してください。
2. 参考文書に書かれていない内容は、絶対に推測や一般知識で補わないでください。
3. 参考文書に答えが見つからない場合は、必ず「該当する情報が見つかりません」と回答してください。それらしい推測の回答は厳禁です。
4. 回答の最後に、根拠にした文書名とセクション名を「出典: ファイル名 / セクション名」の形式で必ず明記してください(複数ある場合は全て列挙)。
5. 回答は簡潔に、2〜3文程度でまとめてください。"""


def answer_question(query, top_k=8):
    results = search(query, top_k=top_k)
    context_parts = []
    for chunk, score in results:
        context_parts.append(
            f"[文書: {chunk['source_file']} / セクション: {chunk['section']}]\n{chunk['content']}"
        )
    context = "\n\n---\n\n".join(context_parts)

    user_prompt = f"""参考文書:
{context}

質問: {query}"""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    answer_text = resp.content[0].text
    return {
        "query": query,
        "retrieved": [(c["source_file"], c["section"], round(s, 3)) for c, s in results],
        "answer": answer_text,
    }


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "有給休暇は入社何ヶ月後に何日付与されますか?"
    result = answer_question(query)
    print("質問:", result["query"])
    print("\n検索結果(上位):")
    for src, sec, score in result["retrieved"]:
        print(f"  - {src} / {sec} (類似度: {score})")
    print("\n回答:")
    print(result["answer"])
