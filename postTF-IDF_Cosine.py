import json
from konlpy.tag import Kkma
from collections import Counter, defaultdict
import math

kkma = Kkma()

def extractRoots(tag):
    # 형태소 분석을 통해 어근만 추출
    return [word for word, pos in kkma.pos(tag) if pos[0] in ['N', 'V', 'M']]  # 체언, 용언의 어간, 수식언

def calculateTF(posts_roots):
    # 게시물별로 TF 계산
    tf = {}
    for post_id, roots in posts_roots.items():
        total_roots = len(roots)
        counts = Counter(roots)
        tf[post_id] = {root: count / total_roots for root, count in counts.items()}
    return tf

def calculateIDF(posts_roots):
    # 전체 게시물에서 특정 어근의 IDF 계산
    num_posts = len(posts_roots)
    idf = defaultdict(float)
    all_roots = set(root for roots in posts_roots.values() for root in roots)
    
    for root in all_roots:
        doc_count = sum(1 for roots in posts_roots.values() if root in roots)
        idf[root] = 1 + math.log(num_posts / (1 + doc_count))
    return idf

def calculateTFIDF(tf, idf):
    # TF와 IDF를 곱해 TF-IDF 계산
    tfidf = {}
    for post_id, tf_values in tf.items():
        tfidf[post_id] = {root: tf_value * idf[root] for root, tf_value in tf_values.items()}
    return tfidf

def cosineSimilarity(vec1, vec2):
    # 두 TF-IDF 벡터 간의 코사인 유사도 계산
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum(vec1[root] * vec2[root] for root in intersection)
    norm1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    norm2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    return numerator / (norm1 * norm2) if norm1 and norm2 else 0

def recommendPosts(input_tags, posts, top_n=5):
    # 게시물 데이터 로드 후 태그를 어근으로 분해
    posts_roots = {post_id: [root for tag in post_tags for root in extractRoots(tag)] for post_id, post_tags in posts.items()}
    input_roots = [root for tag in input_tags for root in extractRoots(tag)]

    # TF, IDF, TF-IDF 계산
    tf = calculateTF(posts_roots)
    idf = calculateIDF(posts_roots)
    tfidf = calculateTFIDF(tf, idf)

    # 입력 태그의 TF-IDF 계산
    input_counts = Counter(input_roots)
    input_tfidf = {root: (count / sum(input_counts.values())) * idf[root] for root, count in input_counts.items()}

    # 코사인 유사도로 추천 게시물 계산
    recommendations = []
    for post_id, tfidf_vec in tfidf.items():
        similarity = cosineSimilarity(input_tfidf, tfidf_vec)
        recommendations.append((post_id, similarity))

    # 유사도를 기준으로 정렬 후 반환
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]

# posts.json 파일에서 데이터 읽기
with open('posts.json', 'r', encoding='utf-8') as f:
    posts = json.load(f)

# 입력 태그 설정
input_tags = ["가족여행", "풍경"]

result = recommendPosts(input_tags, posts, top_n=5)

# 결과 출력
print("추천 게시물:")
for post_id, score in result:
    print(f"게시물 ID {post_id}: 유사도 {score:.2f}")
