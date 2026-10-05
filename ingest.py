"""
案件3 RAGパイプライン - 取り込み(ingest)スクリプト
PDF -> テキスト抽出 -> セクション単位でチャンキング -> Embedding -> Supabaseへ保存
"""
import glob
import json
import os
import pickle
import re
import sys

import pdfplumber
import requests
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SECRET_KEY = os.environ["SUPABASE_SECRET_KEY"]

DOCS_GLOB = "/mnt/user-data/uploads/Downloads/case3-doc*.pdf"

SECTION_RE = re.compile(r"^(第\d+章[^\n]*|^\d+\.\s*[^\n]*)", re.MULTILINE)


def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text


def chunk_by_section(text, source_file):
    """見出し(第N章 / N. )ごとにチャンクへ分割する。500〜1000字程度を目安にしつつ、
    セクションの切れ目を守る(文の途中で切らない)。"""
    matches = list(SECTION_RE.finditer(text))
    chunks = []

    # 見出しより前の文章(タイトル・注記など)
    if matches:
        header_text = text[: matches[0].start()].strip()
    else:
        header_text = text.strip()
    if header_text:
        chunks.append({"section": "(文書冒頭)", "content": header_text})

    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_title = m.group(1).strip()
        body = text[start:end].strip()
        chunks.append({"section": section_title, "content": body})

    # 念のため、1000字を超えるチャンクは段落(改行2つ or 。の後)でさらに分割
    final_chunks = []
    for c in chunks:
        content = c["content"]
        if len(content) <= 1000:
            final_chunks.append(c)
            continue
        parts = re.split(r"(?<=。)\n", content)
        buf = ""
        for p in parts:
            if len(buf) + len(p) > 1000 and buf:
                final_chunks.append({"section": c["section"], "content": buf.strip()})
                buf = ""
            buf += p
        if buf.strip():
            final_chunks.append({"section": c["section"], "content": buf.strip()})

    for c in final_chunks:
        c["source_file"] = source_file
    return final_chunks


def main():
    print("1/4: PDFからテキスト抽出 + チャンキング...")
    all_chunks = []
    for path in sorted(glob.glob(DOCS_GLOB)):
        source_file = os.path.basename(path)
        text = extract_text(path)
        chunks = chunk_by_section(text, source_file)
        print(f"  {source_file}: {len(chunks)} チャンク")
        all_chunks.extend(chunks)
    print(f"  合計 {len(all_chunks)} チャンク\n")

    print("2/4: Embeddingモデルを準備中...")
    # 本来はクラウドの日本語対応Embeddingモデル(多言語モデル等)を使う想定だったが、
    # このサンドボックス環境からはHugging Faceのモデルダウンロードがネットワーク制限で
    # ブロックされたため、代替として文字n-gramベースのTF-IDFベクトルを「Embedding」として使う。
    # (文章を数字の並びに変換し、意味の近さを計算する、という役割は同じ)
    texts = [c["content"] for c in all_chunks]
    # max_featuresで単純に切り捨てると、「接待交際費」のように文書内で1回しか
    # 出てこない重要語がベクトルから欠落してしまう(=検索でヒットしなくなるバグになる)。
    # 対策として、まず語彙を制限せずTF-IDFを計算し、そのあとSVD(次元圧縮)で384次元に
    # 落とし込む。これなら全ての単語の情報を反映したうえで次元数だけ揃えられる。
    # SVDのcomponent数は「サンプル数(チャンク数)-1」が理論上の上限なので、
    # チャンク数が少ない場合はそれより大きい384を指定するとエラーになる。
    # そのため安全に収まる数を計算してから渡す。
    n_components = min(384, len(texts) - 1)
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    vectorizer = make_pipeline(tfidf, svd)
    print("  ロード完了\n")

    print("3/4: 各チャンクをEmbedding(数字の並びに変換)...")
    dense_matrix = vectorizer.fit_transform(texts)
    for c, emb in zip(all_chunks, dense_matrix):
        vec = emb.tolist()
        if len(vec) < 384:
            vec = vec + [0.0] * (384 - len(vec))
        c["embedding"] = vec
    print(f"  {len(all_chunks)}件のEmbedding完了(実効次元数: {n_components} / 保存時は384に0埋め)\n")

    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("4/4: Supabaseへの保存を試行...")
    payload = [
        {
            "content": c["content"],
            "embedding": c["embedding"],
            "source_file": c["source_file"],
            "section": c["section"],
        }
        for c in all_chunks
    ]
    supabase_ok = False
    try:
        requests.delete(
            f"{SUPABASE_URL}/rest/v1/documents",
            headers={
                "apikey": SUPABASE_SECRET_KEY,
                "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
            },
            params={"id": "not.is.null"},
            timeout=10,
        )
        insert_resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/documents",
            headers={
                "apikey": SUPABASE_SECRET_KEY,
                "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=minimal",
            },
            data=json.dumps(payload),
            timeout=10,
        )
        supabase_ok = insert_resp.status_code < 300
        print(f"  挿入: status={insert_resp.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"  Supabaseへの接続に失敗(このサンドボックス環境のネットワーク制限のため): {type(e).__name__}")

    if not supabase_ok:
        print("  → 代わりにローカルに保存して、以降の検索・回答生成はローカルのデータを使います。")
        print("    (本番のGASやNext.jsサーバーから動かす場合は、このSupabase REST APIへの")
        print("     書き込みがそのまま使えます。payloadの形はSupabase用のまま残しています)")

    # 検索・回答生成スクリプトが使えるよう、ローカルに保存(Supabase接続の成否にかかわらず)
    with open("chunks_backup.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n完了! {len(all_chunks)}件のチャンクを保存しました"
          f"({'Supabase' if supabase_ok else 'ローカル'})。")


if __name__ == "__main__":
    main()
