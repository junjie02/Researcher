from dataclasses import dataclass, field
from o2searcher import prompts
import numpy as np
import json
from typing import Union, List, Dict, Any
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class FormatOutput:
    reward: float = 0.0
    metrics: dict = None


def extract_answer(solution_str):
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    if len(matches) < 1:
        return None
    
    return matches[-1].group(1).strip()

def calculate_diversity_reward(content_list, similarity_threshold=0.6, top_k=1):
    if len(content_list) <= 1:
        return 1.0  # Maximum reward for single/empty content
    
    try:
        # Vectorize content using TF-IDF
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(content_list)
        
        # Calculate pairwise cosine similarities
        pairwise_similarity = (tfidf_matrix * tfidf_matrix.T).toarray()
        np.fill_diagonal(pairwise_similarity, 0)  # Zero out self-similarities
        
        # Get upper triangle values (excluding diagonal)
        upper_triangle = pairwise_similarity[np.triu_indices_from(pairwise_similarity, k=1)]
        
        if len(upper_triangle) == 0:
            return 1.0
        
        # New similarity score calculation that better handles few high similarities
        # 1. Count of highly similar pairs (above threshold)
        high_sim_count = np.sum(upper_triangle > similarity_threshold)
        
        # 2. Mean of top-k similarities (handles small clusters)
        top_k_sim = np.mean(np.sort(upper_triangle)[-top_k:]) if len(upper_triangle) >= top_k else np.max(upper_triangle)
        
        # 3. Overall mean similarity
        mean_sim = np.mean(upper_triangle)
        
        # Combine metrics with weights (adjust these based on your needs)
        similarity_score = (
            0.5 * top_k_sim +          # Emphasize cluster similarity
            0.3 * (2 * high_sim_count / len(upper_triangle)) +  # Ratio of high similarities
            0.2 * mean_sim             # Overall similarity
        )
        
        # Transform to diversity reward (higher score = less diversity)
        # Using a non-linear transformation to penalize high similarities more
        diversity_reward = max(0, 1 - np.power(similarity_score, 1.5))
        
        return diversity_reward
        
    except Exception as e:
        print(f"Error calculating diversity reward: {str(e)}")
        return 0.5  # Neutral fallback

def calculate_format_reward(model_answer, data_source='openended'):
    if not model_answer.strip() or model_answer.strip().lower() == 'and':
        return FormatOutput(reward=0.0)

    # close-ended: 不检查 - 格式，有内容就给满分
    if data_source != 'openended':
        return FormatOutput(
            reward=1.0,
            metrics={
                'format': 1.0,
                'completeness': 1.0,
                'diversity': 1.0,
                'duplicate_penalty': 0.0
            }
        )

    # open-ended: 以 "- " 为分隔符切割 bullet，不依赖 \n
    # parts[0] = 第一个 "- " 之前的文本（非空即格式错误）
    # parts[1:] = 每个 bullet 的内容（可能跨多行）
    parts = model_answer.split('- ')
    content_list = []
    format_errors = 0

    if parts[0].strip():
        format_errors += 1

    for part in parts[1:]:
        # 合并换行为空格，压缩空白 → 一行一 bullet
        content = ' '.join(part.split())
        if content:
            content_list.append(content)

    if not content_list:
        return FormatOutput(reward=0.0)

    # open-ended: 检查 - 格式 + 数量 + 多样性
    valid_bullets = len(content_list)
    format_reward = 1 - (format_errors / max(1, valid_bullets + format_errors))
    completeness_reward = min(valid_bullets / 10, 1)
    diversity_reward = calculate_diversity_reward(content_list)

    unique_ratio = len(set(content_list)) / max(1, len(content_list))
    duplicate_penalty = 1 - unique_ratio

    weights = [0.5, 0.3, 0.5]
    reward = (
        (weights[0] * format_reward +
         weights[1] * completeness_reward +
         weights[2] * diversity_reward) / sum(weights) -
        3 * duplicate_penalty
    )
    metrics = {
        'format': weights[0] * format_reward,
        'completeness': weights[1] * completeness_reward,
        'diversity': weights[2] * diversity_reward,
        'duplicate_penalty': -3 * duplicate_penalty
    }

    return FormatOutput(
        reward=max(0, min(1, reward)),
        metrics=metrics
    )

def format_reward_fn(solution_str: str, data_source: str = 'openended'):
    model_answer = extract_answer(solution_str)
    if model_answer is None:
        return FormatOutput(reward=0.0)

    format_reward_output = calculate_format_reward(model_answer, data_source)
    return format_reward_output


def _structure_reward(solution_str: str):
    """Score output structure: presence of <think>, <search>, <answer> tags with content.

    Scans the full solution string (all turns) for structural completeness.
    Content quality is NOT evaluated — that's handled by format_reward_fn and accuracy.

    Returns FormatOutput with reward in [0, 1].
    """
    think_matches = re.findall(r'<think>(.*?)</think>', solution_str, re.DOTALL)
    search_matches = re.findall(r'<search>(.*?)</search>', solution_str, re.DOTALL)
    answer_matches = re.findall(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)

    # think: has tag + content = 1.0, has tag but empty = 0.3, missing = 0.0
    if think_matches:
        has_think_content = any(t.strip() for t in think_matches)
        think_score = 1.0 if has_think_content else 0.3
    else:
        think_score = 0.0

    # search: has tag + content = 1.0, has tag but empty = 0.3, missing = 0.0
    if search_matches:
        has_search_content = any(s.strip() for s in search_matches)
        search_score = 1.0 if has_search_content else 0.3
    else:
        search_score = 0.0

    # answer: has tag + content = 1.0, missing = 0.0
    if answer_matches:
        has_answer_content = any(a.strip() for a in answer_matches)
        answer_score = 1.0 if has_answer_content else 0.0
    else:
        answer_score = 0.0

    weights = [0.3, 0.3, 0.4]  # think, search, answer
    structure_score = weights[0] * think_score + weights[1] * search_score + weights[2] * answer_score

    return FormatOutput(
        reward=float(structure_score),
        metrics={
            'think_structure': think_score,
            'search_structure': search_score,
            'answer_structure': answer_score,
        }
    )


if __name__ == '__main__':
    test_sample = '- ' + '\n- '.join([\
        # "Social media has a double-edged sword effect on adolescent mental health, providing support but also potentially causing negative impacts.",
        "Social media has a double-edged sword effect on adolescent mental health, providing support but also potentially causing negative impacts.",
        "Frequent use of social media may lead to insomnia in adolescents, becoming an accomplice to sleeplessness.",
        "Social media may spread anxiety, affecting adolescents' social anxiety conditions.",
        "The impact of social media on adolescent mental health may be similar to internet addiction, posing potential harm.",
        "The use of social media may pose risks to girls' mental health, possibly related to cyberbullying and lack of sleep.",
        "There is harmful content on social media that needs to be regulated to reduce its negative impact on adolescents.",
        "Research shows that teenagers who use social media for more than 3 hours a day are more likely to experience depressive symptoms, with this risk potentially doubling.",
        # "Reuters reports that the use of social media may severely damage adolescent mental health.",
        "The Ministry of Education warns that social media may have a negative impact on adolescent mental health and needs attention.",
        "Excessive use of social networks may lead to social disorders in adolescents.",
        "Mayo Clinic and other institutions focus on the impact of social media use on adolescents, emphasizing the need to pay attention to its potential risks to mental health.",
        "Current research indicates that the use of social media may have a negative impact on adolescent mental health, but the specific long-term risks are not yet clear.",
        "Hans Publishers and related research point out that there is a certain correlation between the use of social networking sites and adolescent mental health, but more specific cases and detailed information are needed to support these findings.",
        "Despite research progress, there is currently a lack of specific research content on the differences in long-term potential risks of different social media platforms on adolescent mental health.",
        "New York State Attorney General James and other officials advocate for legislation to protect children in response to the long-term mental health risks of social media on adolescents.",
        "Attorneys General from 33 U.S. states are suing Meta, claiming its products harm children's mental health."
    ])
    print(calculate_format_reward(test_sample))