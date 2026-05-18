# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward function for Stock QA dataset with multiple task types (VERSATILE version).
Returns a dict with per-task-type scores and auxiliary metrics for detailed logging.

Task types:
1. pure_forecast: Numerical predictions (prices, percentages)
2. event_detection: Binary Yes/No questions
3. multi_signal_reasoning: Categorical/Strategic answers
4. macro_fundamental: Yes/No with reasoning
5. news_sentiment: Sentiment-based predictions
6. analysis: Long-form analysis

This version includes uncertainty weighting from reward_new.py.
"""



import re
import math
import string
from typing import Union, Dict, Any, List


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_solution(solution_str: str) -> tuple[str, bool]:
    """
    Extract answer from solution string, looking for <answer> tags first.
    Returns (answer, has_answer_tag).
    """
    # remove <|endoftext|>
    solution_str = solution_str.replace("<|endoftext|>", "")
    # Try to extract from <answer> tags
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if matches:
        # Return the last match if multiple exist
        return matches[-1].group(1).strip(), True

    # If no tags, return the entire solution for task-specific extraction
    return solution_str.strip(), False


def extract_number(text: str) -> float:
    """Extract the first number from text, handling percentages."""
    # Remove commas from numbers
    text = text.replace(',', '')

    # Try to find percentage first
    percent_match = re.search(r'(-?\d+\.?\d*)\s*%', text)
    if percent_match:
        return float(percent_match.group(1))

    # Try to find regular number
    number_match = re.search(r'-?\d+\.?\d*', text)
    if number_match:
        return float(number_match.group(0))

    return None


def extract_yes_no(text: str) -> str:
    """Extract Yes/No answer from text."""
    normalized = normalize_answer(text)

    # Check for explicit yes/no at the beginning
    if normalized.startswith('yes'):
        return 'yes'
    elif normalized.startswith('no'):
        return 'no'

    # Check anywhere in text
    if 'yes' in normalized and 'no' not in normalized:
        return 'yes'
    elif 'no' in normalized and 'yes' not in normalized:
        return 'no'

    return None


def extract_day(text: str) -> int:
    """Extract day number from text (e.g., 'Day 2' -> 2)."""
    day_match = re.search(r'day\s*(\d+)', text, re.IGNORECASE)
    if day_match:
        return int(day_match.group(1))
    return None


def check_refusal(text: str) -> bool:
    """Check if the response is a refusal to answer."""
    pred_lower = text.lower()

    refusal_keywords = [
        'cannot predict', 'can\'t predict', 'unable to predict', 'not able to predict',
        'cannot determine', 'can\'t determine', 'unable to determine',
        'insufficient information', 'insufficient data',
        'not enough information', 'not enough data',
        'no indication', 'does not include', 'information provided does not',
        'cannot provide', 'can\'t provide', 'unable to provide',
        'cannot forecast', 'can\'t forecast', 'unable to forecast',
        'impossible to predict', 'impossible to determine',
        'no way to predict', 'no way to determine',
        'cannot be predicted', 'can\'t be predicted',
        'cannot be determined', 'can\'t be determined',
        'unpredictable', 'no prediction', 'decline to predict', 'refuse to predict',
        'without more information', 'without additional data',
        'need more information', 'need more data',
        'require more information', 'require more data'
    ]

    return any(pattern in pred_lower for pattern in refusal_keywords)


def score_pure_forecast(predicted: str, ground_truth: str) -> dict:
    """
    Score numerical predictions using relative error with exponential decay.
    Returns dict with score and additional metrics.
    """
    result = {
        'score': 0.0,
        'is_refusal': 0.0,
        'direction_correct': 0.0,
        'has_prediction': 0.0,
    }

    # Check for refusal patterns first
    if check_refusal(predicted):
        result['is_refusal'] = 1.0
        return result

    # Also check for phrases that indicate uncertainty without providing a number
    uncertainty_phrases = [
        'i don\'t know', 'i do not know', 'it\'s unclear', 'it is unclear',
        'uncertain', 'not sure', 'hard to say', 'difficult to say'
    ]

    pred_lower = predicted.lower()
    if any(phrase in pred_lower for phrase in uncertainty_phrases):
        pred_num = extract_number(predicted)
        if pred_num is None:
            result['is_refusal'] = 1.0
            return result

    # Normal prediction scoring
    pred_num = extract_number(predicted)
    truth_num = extract_number(ground_truth)

    if pred_num is None or truth_num is None:
        return result

    result['has_prediction'] = 1.0

    # Handle zero ground truth
    if abs(truth_num) < 1e-6:
        result['score'] = 1.0 if abs(pred_num) < 1e-6 else 0.0
        result['direction_correct'] = 1.0 if abs(pred_num) < 1e-6 else 0.0
        return result

    # Check direction correctness (sign match or both near zero)
    if (pred_num > 0 and truth_num > 0) or (pred_num < 0 and truth_num < 0) or (abs(pred_num) < 1e-6 and abs(truth_num) < 1e-6):
        result['direction_correct'] = 1.0

    # Calculate relative error
    relative_error = abs(pred_num - truth_num) / abs(truth_num)

    # Exponential decay with alpha = 5
    alpha = 5.0
    score = math.exp(-alpha * relative_error)

    result['score'] = min(1.0, max(0.0, score))
    return result


def score_event_detection(predicted: str, ground_truth: str) -> dict:
    """Score Yes/No questions with binary exact match."""
    result = {
        'score': 0.0,
        'has_valid_answer': 0.0,
    }

    pred_answer = extract_yes_no(predicted)
    truth_answer = extract_yes_no(ground_truth)

    if pred_answer is None or truth_answer is None:
        return result

    result['has_valid_answer'] = 1.0
    result['score'] = 1.0 if pred_answer == truth_answer else 0.0
    return result


def score_multi_signal_reasoning(predicted: str, ground_truth: str) -> dict:
    """Score categorical/strategic answers with hierarchical scoring."""
    result = {
        'score': 0.0,
        'exact_match': 0.0,
        'strategy_match': 0.0,
    }

    pred_answer = extract_yes_no(predicted)
    truth_answer = extract_yes_no(ground_truth)
    if pred_answer is not None and truth_answer is not None:
        result['score'] = 1.0 if pred_answer == truth_answer else 0.0
        result['exact_match'] = 1.0 if pred_answer == truth_answer else 0.0
        return result

    pred_normalized = normalize_answer(predicted)
    truth_normalized = normalize_answer(ground_truth)

    # Check for Day X pattern
    pred_day = extract_day(predicted)
    truth_day = extract_day(ground_truth)

    if pred_day is not None and truth_day is not None:
        day_diff = abs(pred_day - truth_day)
        if day_diff == 0:
            result['score'] = 1.0
            result['exact_match'] = 1.0
        elif day_diff == 1:
            result['score'] = 0.7
        elif day_diff <= 2:
            result['score'] = 0.3
        return result

    # Strategy keywords mapping
    strategy_synonyms = {
        'scalein': ['scale in', 'scalein', 'buy more', 'increase position'],
        'scaleout': ['scale out', 'scaleout', 'sell', 'reduce position', 'decrease position'],
        'stayflat': ['stay flat', 'stayflat', 'hold', 'maintain', 'no change', 'neutral']
    }

    # Find strategies
    truth_strategy = None
    for strategy, keywords in strategy_synonyms.items():
        for keyword in keywords:
            if keyword.replace(' ', '') in truth_normalized.replace(' ', ''):
                truth_strategy = strategy
                break
        if truth_strategy:
            break

    pred_strategy = None
    for strategy, keywords in strategy_synonyms.items():
        for keyword in keywords:
            if keyword.replace(' ', '') in pred_normalized.replace(' ', ''):
                pred_strategy = strategy
                break
        if pred_strategy:
            break

    # Exact match check first
    if pred_normalized == truth_normalized:
        result['score'] = 1.0
        result['exact_match'] = 1.0
        result['strategy_match'] = 1.0
        return result

    # Strategy-based scoring
    if truth_strategy and pred_strategy:
        if truth_strategy == pred_strategy:
            result['score'] = 1.0
            result['strategy_match'] = 1.0
        return result

    # Check for partial keyword matches
    if any(word in pred_normalized for word in truth_normalized.split()):
        result['score'] = 0.5

    return result


def score_macro_fundamental(predicted: str, ground_truth: str) -> dict:
    """Score Yes/No with reasoning."""
    result = {
        'score': 0.0,
        'yes_no_correct': 0.0,
        'reasoning_quality': 0.0,
    }

    pred_answer = extract_yes_no(predicted)
    truth_answer = extract_yes_no(ground_truth)

    if truth_answer is None:
        ground_truth_norm = normalize_answer(ground_truth)
        if ground_truth_norm in ('up', 'down'):
            pred_answer = normalize_answer(predicted)
            if ground_truth_norm in pred_answer:
                result['yes_no_correct'] = 1.0
                base_score = 0.7
            else:
                return result
        else:
            return result
    else:
        if pred_answer is None:
            return result
        # Base score for correct Yes/No
        if pred_answer == truth_answer:
            result['yes_no_correct'] = 1.0
            base_score = 0.7
        else:
            return result

    # Bonus for reasoning quality
    financial_terms = ['cpi', 'volatility', 'cash flow', 'sales', 'baseline',
                      'investing', 'fundamental', 'macro', 'inflation', 'earnings',
                      'revenue', 'profit', 'margin', 'growth', 'decline']

    term_count = sum(1 for term in financial_terms if term in predicted.lower())
    word_count = len(predicted.split())

    reasoning_score = 0.0
    if word_count >= 20 and term_count >= 2:
        reasoning_score = 0.3
        result['reasoning_quality'] = 1.0
    elif word_count >= 10 and term_count >= 1:
        reasoning_score = 0.15
        result['reasoning_quality'] = 0.5

    result['score'] = base_score + reasoning_score
    return result


def score_news_sentiment(predicted: str, ground_truth: str) -> dict:
    """Score sentiment-based predictions."""
    result = {
        'score': 0.0,
        'sentiment_match': 0.0,
    }

    # First try exact Yes/No match
    event_result = score_event_detection(predicted, ground_truth)
    if event_result['score'] == 1.0:
        result['score'] = 1.0
        result['sentiment_match'] = 1.0
        return result

    pred_lower = predicted.lower()
    truth_lower = ground_truth.lower()

    positive_keywords = ['increase', 'rise', 'improve', 'bullish', 'positive', 'recovery', 'higher', 'grow']
    negative_keywords = ['decrease', 'fall', 'decline', 'bearish', 'negative', 'lower', 'drop']
    volatility_keywords = ['volatility', 'volatile', 'fluctuation', 'uncertainty']

    # Check volatility question
    if any(word in pred_lower for word in volatility_keywords):
        if extract_yes_no(ground_truth) == 'yes':
            result['score'] = 0.5
            result['sentiment_match'] = 0.5
            return result

    # Determine sentiment
    pred_sentiment = None
    truth_sentiment = None

    for word in positive_keywords:
        if word in pred_lower:
            pred_sentiment = 'positive'
            break
    for word in negative_keywords:
        if word in pred_lower:
            pred_sentiment = 'negative'
            break

    for word in positive_keywords:
        if word in truth_lower:
            truth_sentiment = 'positive'
            break
    for word in negative_keywords:
        if word in truth_lower:
            truth_sentiment = 'negative'
            break

    if pred_sentiment and truth_sentiment and pred_sentiment == truth_sentiment:
        result['score'] = 0.5
        result['sentiment_match'] = 0.5

    return result


def score_analysis(predicted: str, ground_truth: str) -> dict:
    """Score long-form analysis with multi-criteria evaluation."""
    result = {
        'score': 0.0,
        'has_direction': 0.0,
        'has_key_drivers': 0.0,
        'has_risk': 0.0,
        'has_opportunity': 0.0,
        'is_comprehensive': 0.0,
    }

    pred_lower = predicted.lower()

    # 1. Direction prediction (0.2)
    direction_keywords = {
        'bullish': ['bullish', 'upward', 'positive', 'rise', 'increase', 'up', 'higher'],
        'bearish': ['bearish', 'downward', 'negative', 'fall', 'decrease', 'down', 'lower'],
        'neutral': ['neutral', 'sideways', 'consolidate', 'range', 'flat']
    }

    direction_score = 0.0
    for direction, keywords in direction_keywords.items():
        if any(keyword in pred_lower for keyword in keywords):
            direction_score = 0.2
            result['has_direction'] = 1.0
            break

    # 2. Key drivers mentioned (0.2)
    financial_indicators = ['revenue', 'profit', 'margin', 'growth', 'earnings',
                           'pe ratio', 'debt', 'cash flow', 'eps', 'ebitda',
                           'volume', 'technical', 'fundamental', 'macro']

    indicator_count = sum(1 for indicator in financial_indicators if indicator in pred_lower)
    if indicator_count >= 3:
        key_drivers_score = 0.2
        result['has_key_drivers'] = 1.0
    elif indicator_count >= 2:
        key_drivers_score = 0.1
        result['has_key_drivers'] = 0.5
    else:
        key_drivers_score = 0.0

    # 3. Risk assessment (0.2)
    risk_keywords = ['risk', 'concern', 'challenge', 'threat', 'weakness',
                    'downside', 'volatility', 'uncertainty']

    if any(keyword in pred_lower for keyword in risk_keywords):
        risk_score = 0.2
        result['has_risk'] = 1.0
    else:
        risk_score = 0.0

    # 4. Opportunity identification (0.2)
    opportunity_keywords = ['opportunity', 'potential', 'upside', 'strength',
                           'advantage', 'growth', 'catalyst', 'positive']

    if any(keyword in pred_lower for keyword in opportunity_keywords):
        opportunity_score = 0.2
        result['has_opportunity'] = 1.0
    else:
        opportunity_score = 0.0

    # 5. Comprehensiveness (0.2)
    word_count = len(predicted.split())
    if word_count >= 200:
        comprehensiveness_score = 0.2
        result['is_comprehensive'] = 1.0
    elif word_count >= 100:
        comprehensiveness_score = 0.1
        result['is_comprehensive'] = 0.5
    elif word_count >= 50:
        comprehensiveness_score = 0.05
        result['is_comprehensive'] = 0.25
    else:
        comprehensiveness_score = 0.0

    result['score'] = min(1.0, max(0.0, direction_score + key_drivers_score + risk_score + opportunity_score + comprehensiveness_score))
    return result


def uncertainty_weight(
    u: float,
    u0: float = 0.25,
    u_hi: float = 0.50,
    w_hi: float = 0.50,
    u_cap: float = 1.0,
) -> float:
    """
    Map data uncertainty u -> weight in (0, 1].
    Smaller weight => less influence on training.
    """

    if u < 0.0:
        u = 0.0
    if u > u_cap:
        u = u_cap
    if u <= u0:
        return 1.0
    denom = max(1e-8, (u_hi - u0))

    w_hi = min(max(w_hi, 1e-6), 1.0)
    alpha = -math.log(w_hi)

    t = (u - u0) / denom
    w = math.exp(-alpha * t)

    if w < 0.0:
        w = 0.0
    if w > 1.0:
        w = 1.0
    return w


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Main scoring function for Stock QA dataset (VERSATILE version).
    Returns a dict with 'score' and per-task-type auxiliary metrics.

    Args:
        data_source: Data source identifier (required by VERL interface)
        solution_str: The model's response (detokenized string)
        ground_truth: Ground truth answer or dict with 'target' field
        extra_info: Additional information including 'task' field

    Returns:
        Dict with 'score' (main reward) and auxiliary metrics for logging.
        All auxiliary metrics will be logged under val-aux/{data_source}/{metric}/mean@N
    """
    # Initialize result dict with all metrics set to 0
    result = {
        'score': 0.0,
        # Format metrics
        'has_answer_tag': 0.0,
        'response_length': 0.0,
        # Per-task-type scores (only one will be non-zero per sample)
        'pure_forecast': 0.0,
        'event_detection': 0.0,
        'multi_signal_reasoning': 0.0,
        'macro_fundamental': 0.0,
        'news_sentiment': 0.0,
        'analysis': 0.0,
        'fallback': 0.0,
        # Additional metrics
        'is_refusal': 0.0,
        'direction_correct': 0.0,
        'exact_match': 0.0,
    }

    # Extract answer from solution
    answer, has_answer_tag = extract_solution(solution_str)
    result['has_answer_tag'] = 1.0 if has_answer_tag else 0.0
    result['response_length'] = float(len(solution_str.split()))

    if answer is None or answer == "":
        return result

    # Handle ground truth format
    if isinstance(ground_truth, dict):
        if 'target' in ground_truth:
            targets = ground_truth['target']
            if isinstance(targets, list) and len(targets) > 0:
                ground_truth_str = targets[0]
            else:
                ground_truth_str = str(targets)
        elif 'answer' in ground_truth:
            ground_truth_str = str(ground_truth['answer'])
        else:
            ground_truth_str = str(ground_truth)
    else:
        ground_truth_str = str(ground_truth)

    # Get task type from extra_info
    task_type = None
    if extra_info and isinstance(extra_info, dict):
        task_type = extra_info.get('question_type', None)

    # Route to appropriate scoring function based on task type
    if task_type == 'pure_forecast':
        task_result = score_pure_forecast(answer, ground_truth_str)
        result['score'] = task_result['score']
        result['pure_forecast'] = task_result['score']
        result['is_refusal'] = task_result.get('is_refusal', 0.0)
        result['direction_correct'] = task_result.get('direction_correct', 0.0)

    elif task_type == 'event_detection':
        task_result = score_event_detection(answer, ground_truth_str)
        result['score'] = task_result['score']
        result['event_detection'] = task_result['score']
        result['exact_match'] = task_result['score']

    elif task_type == 'multi_signal_reasoning':
        task_result = score_multi_signal_reasoning(answer, ground_truth_str)
        result['score'] = task_result['score']
        result['multi_signal_reasoning'] = task_result['score']
        result['exact_match'] = task_result.get('exact_match', 0.0)

    elif task_type == 'macro_fundamental':
        task_result = score_macro_fundamental(answer, ground_truth_str)
        result['score'] = task_result['score']
        result['macro_fundamental'] = task_result['score']
        result['exact_match'] = task_result.get('yes_no_correct', 0.0)

    elif task_type == 'news_sentiment':
        task_result = score_news_sentiment(answer, ground_truth_str)
        result['score'] = task_result['score']
        result['news_sentiment'] = task_result['score']

    elif task_type == 'analysis':
        task_result = score_analysis(answer, ground_truth_str)
        result['score'] = task_result['score']
        result['analysis'] = task_result['score']

    else:
        # Fallback: generic exact match
        normalized_pred = normalize_answer(answer)
        normalized_truth = normalize_answer(ground_truth_str)

        if normalized_pred == normalized_truth:
            score = 1.0
            result['exact_match'] = 1.0
        elif any(word in normalized_pred for word in normalized_truth.split()):
            score = 0.5
        else:
            score = 0.0
        result['score'] = score
        result['fallback'] = score

    # Apply uncertainty weighting if available (this is the "new" part)
    if extra_info and isinstance(extra_info, dict) and "uncertainty" in extra_info:
        uncertainty_w = uncertainty_weight(extra_info["uncertainty"], u0=0.30, u_hi=0.50, w_hi=0.50, u_cap=1.0)
        result['score'] = result['score'] * uncertainty_w

    result['score'] = min(1.0, max(0.0, result['score']))
    return result


