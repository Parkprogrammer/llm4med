from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np

# 추천 교육자료 Matrix 생성
def build_tfidf_index(content_list):
    """
    content_list: [
      {
        "No": 1,
        "Title": "...",
        "File_name": "...",
        "Keywords": ["당뇨", "당뇨병", "당화혈색소", ...]
      },
      ...
    ]

    1) 각 항목의 '전체키워드'를 공백으로 join → 하나의 '문서' 형태
    2) TfidfVectorizer 로 fit_transform → (n_docs x n_features) 행렬 X
    3) vectorizer, X, content_list 반환
    """
    corpus = []
    for content in content_list:
        # ['당뇨', '당뇨병', '당화혈색소'] → "당뇨 당뇨병 당화혈색소"
        doc = " ".join(k.strip().lower() for k in content["전체키워드"])
        corpus.append(doc)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)  # (n_docs x n_features) 희소행렬
    return vectorizer, X, content_list


# 키워드 추출 함수
def extract_keywords_from_output(output_text):
    """키워드 추출 함수"""
    # 1) 먼저 각 줄을 분리
    lines = output_text.split('\n')
    keywords = []

    for line in lines:
        # 2) '**' 사이의 내용을 추출 (번호나 다른 포맷에 상관없이)
        matches = re.findall(r'\*\*(.*?)\*\*', line)
        keywords.extend([m.strip() for m in matches if m.strip()])

    print("Extracted keywords:", keywords)  # 디버깅용
    return keywords

# NOTE: 해당 top_n 파라미터가 추천 될 교육자료의 개수 선정
def find_best_matches_tfidf(extracted_keywords, vectorizer, X, content_list, top_n=5):
    """
    extracted_keywords: ["당뇨", "식단", "운동", ...] (Stage3에서 추출된 키워드)
    vectorizer: 위에서 생성한 TfidfVectorizer
    X: (n_docs x n_features) TF-IDF 행렬 -> build_tfidf_index
    content_list: 문서 메타정보(파일명, 제목, 전체키워드 등)
    top_n: 상위 몇 개를 뽑을지 (기본 5) : 주요 파라미터
    """
    # HACK: 임신과 같은 핵심키워드에 대해 좀 더 파라미터를 강조함.
    has_pregnancy_keyword = any("임신" in keyword for keyword in extracted_keywords)

    query_str = " ".join(k.strip().lower() for k in extracted_keywords)
    query_vec = vectorizer.transform([query_str])  # (1 x n_features)

    if query_vec.nnz == 0:
        return []

    sim_scores = cosine_similarity(X, query_vec).flatten()  # 1D array

    if has_pregnancy_keyword:
        for idx, content in enumerate(content_list):
            if any("임신" in keyword for keyword in content["전체키워드"]) or "임신" in content["제목"]:
                sim_scores[idx] *= 100.0

    ranked_indices = np.argsort(-sim_scores)
    top_indices = ranked_indices[:top_n]

    results = []
    for idx in top_indices:
        score = sim_scores[idx]
        if score > 0:
            results.append((score, content_list[idx]))

    return results # [(유사도, {...}), ...]