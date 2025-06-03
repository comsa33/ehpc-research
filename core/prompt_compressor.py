import logging
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from .evaluator_heads import CompressionResult, EvaluatorHeadFinder, EvaluatorHeadInfo


class EHPCCompressor:
    """EHPC 프롬프트 압축기 - 논문의 메인 알고리즘 구현"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.head_finder = EvaluatorHeadFinder(model_name)
        self.evaluator_heads: Optional[List[EvaluatorHeadInfo]] = None
        self.is_initialized = False

    def initialize(
        self, max_layers: int = 3, heads_per_layer: int = 2
    ) -> List[EvaluatorHeadInfo]:
        """시스템 초기화 - Evaluator Heads 찾기"""
        logging.info("Initializing EHPC compressor...")
        self.evaluator_heads = self.head_finder.find_evaluator_heads(
            max_layers=max_layers, heads_per_layer=heads_per_layer
        )
        self.is_initialized = True
        return self.evaluator_heads

    def compress_prompt(
        self,
        text: str,
        compression_ratio: float = 0.3,
        preserve_special_tokens: bool = True,
    ) -> CompressionResult:
        """
        프롬프트 압축 메인 함수

        Args:
            text: 압축할 텍스트
            compression_ratio: 유지할 토큰 비율 (0.3 = 30% 유지)
            preserve_special_tokens: 특수 토큰 보존 여부
        """
        if not self.is_initialized:
            raise ValueError(
                "compress_prompt를 호출하기 전에 initialize()를 먼저 호출해야 합니다."
            )

        # 토큰화
        inputs = self.head_finder.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        )

        tokens = self.head_finder.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

        # Evaluator heads의 attention 점수 계산
        with torch.no_grad():
            outputs = self.head_finder.model(**inputs)
            attentions = outputs.attentions

        # 각 토큰의 중요도 점수 계산
        token_scores = self._calculate_token_importance(attentions, tokens)

        # 압축할 토큰 수 결정
        target_length = max(int(len(tokens) * compression_ratio), 5)

        # 중요한 토큰들 선택
        selected_indices = self._select_important_tokens(
            token_scores, target_length, tokens, preserve_special_tokens
        )

        # 압축된 토큰들
        compressed_tokens = [tokens[i] for i in selected_indices]

        return CompressionResult(
            original_tokens=tokens,
            compressed_tokens=compressed_tokens,
            token_scores=token_scores,
            selected_indices=selected_indices,
            compression_ratio=len(compressed_tokens) / len(tokens),
            evaluator_heads=self.evaluator_heads,
        )

    def _calculate_token_importance(
        self, attentions: tuple, tokens: List[str]
    ) -> np.ndarray:
        """Evaluator heads의 attention을 바탕으로 토큰 중요도 계산"""
        num_tokens = len(tokens)
        importance_scores = np.zeros(num_tokens)

        for head_info in self.evaluator_heads:
            layer_idx = head_info.layer
            head_idx = head_info.head

            if layer_idx < len(attentions):
                # 해당 헤드의 attention 가져오기
                attention = attentions[layer_idx][0, head_idx]  # [seq_len, seq_len]

                # 방법 1: 각 토큰이 받는 전체 attention (다른 토큰들로부터)
                incoming_attention = attention.sum(dim=0).cpu().numpy()

                # 방법 2: 각 토큰이 주는 attention의 집중도 (entropy 역수)
                outgoing_attention = attention.sum(dim=1).cpu().numpy()

                # 두 점수를 결합 (가중 평균)
                combined_score = 0.7 * incoming_attention + 0.3 * outgoing_attention

                # 헤드의 신뢰도로 가중
                weighted_score = combined_score * head_info.confidence_score
                importance_scores += weighted_score

        # 정규화
        if importance_scores.max() > 0:
            importance_scores = importance_scores / importance_scores.max()

        return importance_scores

    def _select_important_tokens(
        self,
        scores: np.ndarray,
        target_length: int,
        tokens: List[str],
        preserve_special_tokens: bool,
    ) -> List[int]:
        """중요도 점수를 바탕으로 토큰 선택"""

        # 특수 토큰 식별
        special_token_indices = set()
        if preserve_special_tokens:
            for i, token in enumerate(tokens):
                if (token.startswith("<") and token.endswith(">")) or token in [
                    "[CLS]",
                    "[SEP]",
                    "[PAD]",
                    "<s>",
                    "</s>",
                    "<|endoftext|>",
                ]:
                    special_token_indices.add(i)

        # 특수 토큰은 항상 포함
        selected_indices = list(special_token_indices)
        remaining_slots = target_length - len(selected_indices)

        if remaining_slots > 0:
            # 특수 토큰이 아닌 토큰들 중에서 높은 점수 순으로 선택
            candidates = [
                (i, scores[i])
                for i in range(len(scores))
                if i not in special_token_indices
            ]

            # 점수 순으로 정렬
            candidates.sort(key=lambda x: x[1], reverse=True)

            # 상위 토큰들 선택
            for i, (token_idx, score) in enumerate(candidates[:remaining_slots]):
                selected_indices.append(token_idx)

        # 원래 순서대로 정렬
        selected_indices.sort()
        return selected_indices

    def tokens_to_text(self, tokens: List[str]) -> str:
        """토큰 리스트를 자연스러운 텍스트로 변환"""
        text = self.head_finder.tokenizer.convert_tokens_to_string(tokens)

        # 간단한 후처리로 자연스러운 텍스트 만들기
        text = re.sub(r"\s+", " ", text)  # 중복 공백 제거
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # 구두점 앞 공백 제거

        return text.strip()

    def compress_and_generate(
        self,
        text: str,
        compression_ratio: float = 0.3,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
    ) -> Dict[str, Union[str, float, int]]:
        """
        프롬프트를 압축하고 같은 모델로 생성까지 수행
        원본과 압축 버전의 결과를 비교
        """
        # 1. 원본으로 생성
        original_inputs = self.head_finder.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024
        )

        with torch.no_grad():
            original_outputs = self.head_finder.model.generate(
                **original_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.head_finder.tokenizer.eos_token_id,
            )

        original_response = self.head_finder.tokenizer.decode(
            original_outputs[0], skip_special_tokens=True
        )

        # 2. 압축 후 생성
        compression_result = self.compress_prompt(text, compression_ratio)
        compressed_text = self.tokens_to_text(compression_result.compressed_tokens)

        compressed_inputs = self.head_finder.tokenizer(
            compressed_text, return_tensors="pt", truncation=True, max_length=1024
        )

        with torch.no_grad():
            compressed_outputs = self.head_finder.model.generate(
                **compressed_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.head_finder.tokenizer.eos_token_id,
            )

        compressed_response = self.head_finder.tokenizer.decode(
            compressed_outputs[0], skip_special_tokens=True
        )

        return {
            "original_text": text,
            "compressed_text": compressed_text,
            "original_response": original_response,
            "compressed_response": compressed_response,
            "compression_ratio": compression_result.compression_ratio,
            "original_length": len(text.split()),
            "compressed_length": len(compressed_text.split()),
            "tokens_kept": len(compression_result.selected_indices),
            "tokens_total": len(compression_result.original_tokens),
        }
