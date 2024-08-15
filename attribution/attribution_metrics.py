import math
import re
from typing import cast

import numpy as np
import torch
from openai.types.chat.chat_completion_token_logprob import TopLogprob
from sklearn.metrics.pairwise import cosine_similarity
from transformers import PreTrainedTokenizer

from .types import StrictChoiceLogprobs
from sentence_transformers import SentenceTransformer, util

NEAR_ZERO_PROB = -100  # Logprob constant for near zero probability


def token_prob_attribution(
        initial_logprobs: StrictChoiceLogprobs,
        perturbed_logprobs: StrictChoiceLogprobs,
) -> dict[str, float]:
    # Extract token and logprob from initial_logprobs
    initial_token_logprobs = [
        (logprob.token, logprob.logprob) for logprob in initial_logprobs.content
    ]

    # Create a list of dictionaries with token and top logprobs from perturbed_logprobs
    perturbed_token_logprobs_list = [
        {top_logprob.token: top_logprob.logprob for top_logprob in token_content.top_logprobs}
        for token_content in perturbed_logprobs.content
    ]

    # Probability change for each input token
    prob_difference_per_token = {}

    # Calculate the absolute difference in probabilities for each token
    for i, initial_token in enumerate(initial_token_logprobs):
        perturbed_token_logprobs = (
            perturbed_token_logprobs_list[i] if i < len(perturbed_token_logprobs_list) else {}
        )
        perturbed_logprob = perturbed_token_logprobs.get(initial_token[0], NEAR_ZERO_PROB)
        prob_difference = math.exp(initial_token[1]) - math.exp(perturbed_logprob)
        prob_difference_per_token[f"{initial_token[0]} ({i})"] = prob_difference

    return prob_difference_per_token


def cosine_similarity_attribution(
        original_output_str: str,
        perturbed_output_str: str,
        token_embeddings: np.ndarray,
        tokenizer: PreTrainedTokenizer,
) -> dict[str, float]:
    # Extract embeddings
    original_token_id = cast(
        torch.Tensor,
        tokenizer.encode(original_output_str, return_tensors="pt", add_special_tokens=False),
    )
    perturbed_token_id = cast(
        torch.Tensor,
        tokenizer.encode(perturbed_output_str, return_tensors="pt", add_special_tokens=False),
    )
    initial_tokens = [tokenizer.decode(t) for t in original_token_id.squeeze(dim=0)]

    original_output_emb = token_embeddings[original_token_id].reshape(
        -1, token_embeddings.shape[-1]
    )
    perturbed_output_emb = token_embeddings[perturbed_token_id].reshape(
        -1, token_embeddings.shape[-1]
    )

    cosine_distance = 1 - cosine_similarity(original_output_emb, perturbed_output_emb)
    token_distance = cosine_distance.min(axis=-1)
    return {
        token + f" {i}": distance
        for i, (token, distance) in enumerate(zip(initial_tokens, token_distance))
    }


def _is_token_in_top_20(
        token: str,
        top_logprobs: list[TopLogprob],
):
    top_20_tokens = set(logprob.token for logprob in top_logprobs)
    return token in top_20_tokens


def any_tokens_in_top_20(
        initial_logprobs: StrictChoiceLogprobs,
        new_logprobs: StrictChoiceLogprobs,
) -> bool:
    return any(
        _is_token_in_top_20(initial_token.token, new_token.top_logprobs)
        for initial_token, new_token in zip(initial_logprobs.content, new_logprobs.content)
    )


def target_similarity_attribution(
        original_output: str,
        perturbed_output: str,
        target_output: str,
        tokenizer: PreTrainedTokenizer,
        token_embeddings: np.ndarray
) -> dict[str, float]:
    original_tokens = tokenizer.encode(original_output, add_special_tokens=False)
    perturbed_tokens = tokenizer.encode(perturbed_output, add_special_tokens=False)
    target_tokens = tokenizer.encode(target_output, add_special_tokens=False)

    original_embeddings = token_embeddings[original_tokens]
    perturbed_embeddings = token_embeddings[perturbed_tokens]
    target_embeddings = token_embeddings[target_tokens]

    original_target_sim = cosine_similarity(original_embeddings, target_embeddings)
    perturbed_target_sim = cosine_similarity(perturbed_embeddings, target_embeddings)

    attribution_scores = {}

    for i, original_token in enumerate(original_tokens):
        most_similar_target_idx = original_target_sim[i].argmax()

        original_similarity = original_target_sim[i, most_similar_target_idx]

        perturbed_similarity = perturbed_target_sim[:, most_similar_target_idx].max()

        similarity_change = original_similarity - perturbed_similarity

        token_str = tokenizer.decode([original_token])
        attribution_scores[f"{token_str} ({i})"] = similarity_change

    return attribution_scores


def sentence_similarity(text1: str, text2: str) -> float:
    st_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding1 = st_model.encode(text1, convert_to_tensor=True)
    embedding2 = st_model.encode(text2, convert_to_tensor=True)
    return cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))[0][0].item()


async def sentence_similarity_llm(text1: str, text2: str, openai_client) -> float:
    def extract_float_from_tags(input_string):
        pattern = r'<rating>([-+]?\d*\.?\d+)</rating>'
        match = re.search(pattern, input_string)

        if match:
            return float(match.group(1))
        else:
            return None

    prompt = f"""
    Please rate between 0 and 1 how similar in meaning the following texts are. Respond only with the rating, contained within tags like so: <rating>0.2</rating>. 
    <text1>
    {text1}
    </text1>
    
    <text2>
    {text2}
    </text2>
    """
    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        seed=0,
    )
    rating = extract_float_from_tags(response.choices[0].message.content)
    return rating


async def target_similarity_attribution_llm(
        original_output: str,
        perturbed_output: str,
        target_output: str,
        tokenizer: PreTrainedTokenizer,
        token_embeddings: np.ndarray,
        openai_client,
) -> dict[str, float]:
    original_target_sim = await sentence_similarity_llm(original_output, target_output, openai_client)
    perturbed_target_sim = await sentence_similarity_llm(perturbed_output, target_output, openai_client)

    similarity_change = original_target_sim - perturbed_target_sim

    original_tokens = tokenizer.encode(original_output, add_special_tokens=False)

    attribution_scores = {}
    for i, token_id in enumerate(original_tokens):
        token_str = tokenizer.decode([token_id])
        attribution_scores[f"{token_str} ({i})"] = similarity_change

    return attribution_scores


def target_similarity_attribution_st(
        original_output: str,
        perturbed_output: str,
        target_output: str,
        tokenizer: PreTrainedTokenizer,
        token_embeddings: np.ndarray,
) -> dict[str, float]:
    original_target_sim = sentence_similarity(original_output, target_output)
    perturbed_target_sim = sentence_similarity(perturbed_output, target_output)

    similarity_change = original_target_sim - perturbed_target_sim

    original_tokens = tokenizer.encode(original_output, add_special_tokens=False)

    attribution_scores = {}
    for i, token_id in enumerate(original_tokens):
        token_str = tokenizer.decode([token_id])
        attribution_scores[f"{token_str} ({i})"] = similarity_change

    return attribution_scores
