import logging
import re
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from core.evaluator_heads import (
    CompressionResult,
    EvaluatorHeadFinder,
    EvaluatorHeadInfo,
    get_recommended_model,
    safe_tokenize,
)


class EHPCCompressor:
    """EHPC 프롬프트 압축기 - 논문의 메인 알고리즘 구현 (업그레이드 버전)"""

    def __init__(self, model_name: Optional[str] = None, auto_initialize: bool = False):
        """
        Args:
            model_name: 사용할 모델 이름 (None이면 자동 선택)
            auto_initialize: True면 생성 시 바로 초기화
        """
        if model_name is None:
            model_name = get_recommended_model()
            logging.info(f"🎯 자동 선택된 모델: {model_name}")

        self.model_name = model_name
        self.head_finder = EvaluatorHeadFinder(model_name)
        self.evaluator_heads: Optional[List[EvaluatorHeadInfo]] = None
        self.is_initialized = False

        # 모델 정보 로깅
        model_info = self.head_finder.get_model_info_dict()
        logging.info(f"📚 로드된 모델 정보: {model_info}")

        if auto_initialize:
            self.initialize()

    def initialize(
        self,
        max_layers: int = 3,
        heads_per_layer: int = 2,
        force_reinitialize: bool = False,
    ) -> List[EvaluatorHeadInfo]:
        """
        시스템 초기화 - Evaluator Heads 찾기

        Args:
            max_layers: 검사할 최대 레이어 수
            heads_per_layer: 레이어당 선택할 헤드 수
            force_reinitialize: 이미 초기화된 경우에도 강제로 재초기화
        """
        if self.is_initialized and not force_reinitialize:
            logging.info("⚡ 이미 초기화됨, 기존 결과 사용")
            return self.evaluator_heads

        logging.info("🚀 EHPC 압축기 초기화 시작...")

        # 모델 크기에 따른 적응적 설정
        model_config = self.head_finder.model_config
        params = model_config.get("params", "1B")

        # 작은 모델의 경우 더 많은 레이어 검사
        if "2B" in params or "354M" in params:
            max_layers = min(max_layers, 4)
            heads_per_layer = max(heads_per_layer, 2)
        elif "3B" in params or "5B" in params:
            max_layers = min(max_layers, 3)
            heads_per_layer = max(heads_per_layer, 2)

        logging.info(
            f"📊 설정: max_layers={max_layers}, heads_per_layer={heads_per_layer}"
        )

        try:
            self.evaluator_heads = self.head_finder.find_evaluator_heads(
                max_layers=max_layers, heads_per_layer=heads_per_layer
            )
            self.is_initialized = True

            logging.info(
                f"✅ 초기화 완료: {len(self.evaluator_heads)}개 Evaluator Heads 발견"
            )
            return self.evaluator_heads

        except Exception as e:
            logging.error(f"❌ 초기화 실패: {e}")
            # 백업 헤드 사용
            self.evaluator_heads = [
                EvaluatorHeadInfo(
                    layer=0, head=0, selectivity_score=0.5, confidence_score=0.1
                )
            ]
            self.is_initialized = True
            logging.warning("⚠️ 백업 헤드로 초기화")
            return self.evaluator_heads

    def compress_prompt(
        self,
        text: str,
        compression_ratio: float = 0.3,
        preserve_special_tokens: bool = True,
        min_tokens: int = 5,
        max_tokens: Optional[int] = None,
    ) -> CompressionResult:
        """
        프롬프트 압축 메인 함수 (개선된 버전)

        Args:
            text: 압축할 텍스트
            compression_ratio: 유지할 토큰 비율 (0.3 = 30% 유지)
            preserve_special_tokens: 특수 토큰 보존 여부
            min_tokens: 최소 유지 토큰 수
            max_tokens: 최대 토큰 수 제한
        """
        if not self.is_initialized:
            logging.info("🔧 자동 초기화 수행...")
            self.initialize()

        # 토큰화 및 길이 제한
        model_max_length = self.head_finder.model_config.get("context_length", 2048)
        effective_max_length = min(model_max_length, max_tokens or 2048)

        inputs = safe_tokenize(
            self.head_finder.tokenizer,
            text,
            model_name=self.model_name,
            model_config=self.head_finder.model_config,
            max_length=effective_max_length,
        )

        # 디바이스로 이동
        inputs = {k: v.to(self.head_finder.device) for k, v in inputs.items()}
        tokens = self.head_finder.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0]
        )

        if len(tokens) == 0:
            raise ValueError("❌ 토큰화 결과가 비어있습니다.")

        # Attention 계산
        with torch.no_grad():
            try:
                outputs = self.head_finder.model(**inputs)
                attentions = outputs.attentions
            except Exception as e:
                logging.error(f"❌ 모델 추론 실패: {e}")
                # 폴백: 균등 분포 점수 사용
                token_scores = np.ones(len(tokens)) / len(tokens)
                logging.warning("⚠️ 폴백: 균등 분포 점수 사용")
            else:
                # 토큰 중요도 계산
                token_scores = self._calculate_token_importance(attentions, tokens)

        # 압축할 토큰 수 결정
        target_length = max(
            int(len(tokens) * compression_ratio),
            min_tokens,
            min(len(tokens), 3),  # 최소한 3개는 유지
        )

        if max_tokens and target_length > max_tokens:
            target_length = max_tokens

        # 중요한 토큰들 선택
        selected_indices = self._select_important_tokens(
            token_scores, target_length, tokens, preserve_special_tokens
        )

        # 압축된 토큰들
        compressed_tokens = [tokens[i] for i in selected_indices]
        actual_compression_ratio = len(compressed_tokens) / len(tokens)

        logging.info(
            f"📊 압축 결과: {len(tokens)} → {len(compressed_tokens)} 토큰 "
            f"(목표: {compression_ratio:.1%}, 실제: {actual_compression_ratio:.1%})"
        )

        return CompressionResult(
            original_tokens=tokens,
            compressed_tokens=compressed_tokens,
            token_scores=token_scores,
            selected_indices=selected_indices,
            compression_ratio=actual_compression_ratio,
            evaluator_heads=self.evaluator_heads,
        )

    def _calculate_token_importance(
        self, attentions: tuple, tokens: List[str]
    ) -> np.ndarray:
        """Evaluator heads의 attention을 바탕으로 토큰 중요도 계산 (개선된 버전)"""
        num_tokens = len(tokens)
        importance_scores = np.zeros(num_tokens)

        if not self.evaluator_heads:
            logging.warning("⚠️ Evaluator heads가 없음, 균등 분포 반환")
            return np.ones(num_tokens) / num_tokens

        total_weight = 0
        for head_info in self.evaluator_heads:
            layer_idx = head_info.layer
            head_idx = head_info.head

            if layer_idx < len(attentions):
                try:
                    # 해당 헤드의 attention 가져오기
                    attention = attentions[layer_idx][0, head_idx]  # [seq_len, seq_len]

                    # 크기 검증
                    if (
                        attention.size(0) != num_tokens
                        or attention.size(1) != num_tokens
                    ):
                        logging.debug(
                            f"Attention 크기 불일치: {attention.shape} vs {num_tokens}"
                        )
                        continue

                    # 개선된 중요도 계산
                    # 1. 각 토큰이 받는 attention (다른 토큰들로부터)
                    incoming_attention = attention.sum(dim=0).cpu().numpy()

                    # 2. 각 토큰이 주는 attention의 분산 (선택성)
                    outgoing_attention = attention.sum(dim=1).cpu().numpy()

                    # 3. Self-attention 가중치 (대각선 요소)
                    self_attention = torch.diag(attention).cpu().numpy()

                    # 4. 가중 결합 (더 정교한 공식)
                    combined_score = (
                        0.5 * incoming_attention
                        + 0.3 * outgoing_attention
                        + 0.2 * self_attention
                    )

                    # 헤드의 신뢰도로 가중
                    head_weight = head_info.confidence_score
                    weighted_score = combined_score * head_weight

                    importance_scores += weighted_score
                    total_weight += head_weight

                except Exception as e:
                    logging.debug(f"헤드 {layer_idx}-{head_idx} 처리 중 오류: {e}")
                    continue

        # 정규화
        if total_weight > 0:
            importance_scores = importance_scores / total_weight

        if importance_scores.max() > 0:
            importance_scores = importance_scores / importance_scores.max()
        else:
            # 폴백: 위치 기반 중요도 (시작과 끝 강조)
            importance_scores = self._get_positional_importance(num_tokens)
            logging.warning("⚠️ Attention 기반 점수 계산 실패, 위치 기반 폴백 사용")

        return importance_scores

    def _get_positional_importance(self, num_tokens: int) -> np.ndarray:
        """위치 기반 중요도 점수 (폴백용)"""
        scores = np.ones(num_tokens)

        # 시작 토큰들에 높은 가중치
        start_boost = min(3, num_tokens // 4)
        scores[:start_boost] *= 1.5

        # 끝 토큰들에 높은 가중치
        end_boost = min(2, num_tokens // 6)
        if end_boost > 0:
            scores[-end_boost:] *= 1.3

        return scores / scores.sum()

    def _select_important_tokens(
        self,
        scores: np.ndarray,
        target_length: int,
        tokens: List[str],
        preserve_special_tokens: bool,
    ) -> List[int]:
        """중요도 점수를 바탕으로 토큰 선택 (개선된 버전)"""

        # 특수 토큰 식별 (확장된 리스트)
        special_token_indices = set()
        if preserve_special_tokens:
            special_patterns = [
                r"^<[^>]*>$",  # <start>, <end> 등
                r"^\[[^\]]*\]$",  # [CLS], [SEP] 등
                r"^▁",  # SentencePiece prefix
            ]

            special_tokens = {
                "[CLS]",
                "[SEP]",
                "[PAD]",
                "[UNK]",
                "[MASK]",
                "<s>",
                "</s>",
                "<pad>",
                "<unk>",
                "<|endoftext|>",
                "Ġ",  # GPT-2 space token
            }

            for i, token in enumerate(tokens):
                # 패턴 매칭
                if any(re.match(pattern, token) for pattern in special_patterns):
                    special_token_indices.add(i)
                # 직접 매칭
                elif token in special_tokens:
                    special_token_indices.add(i)
                # 첫 번째와 마지막 토큰 보존
                elif i == 0 or i == len(tokens) - 1:
                    special_token_indices.add(i)

        # 특수 토큰은 항상 포함
        selected_indices = list(special_token_indices)
        remaining_slots = max(0, target_length - len(selected_indices))

        if remaining_slots > 0:
            # 일반 토큰들 중에서 선택
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

        # 연속성 확보를 위한 후처리
        selected_indices = self._ensure_continuity(selected_indices, tokens)

        return selected_indices

    def _ensure_continuity(self, indices: List[int], tokens: List[str]) -> List[int]:
        """텍스트 연속성을 위한 인덱스 조정"""
        if len(indices) <= 3:
            return indices

        # 너무 큰 간격이 있는 경우 중간 토큰 추가
        final_indices = []
        prev_idx = indices[0]
        final_indices.append(prev_idx)

        for curr_idx in indices[1:]:
            gap = curr_idx - prev_idx

            # 간격이 너무 크면 중간에 토큰 하나 추가
            if gap > 5 and len(final_indices) < len(indices) * 1.2:
                mid_idx = (prev_idx + curr_idx) // 2
                if mid_idx not in final_indices:
                    final_indices.append(mid_idx)

            final_indices.append(curr_idx)
            prev_idx = curr_idx

        return sorted(list(set(final_indices)))

    def tokens_to_text(self, tokens: List[str]) -> str:
        """토큰 리스트를 자연스러운 텍스트로 변환 (개선된 버전)"""
        if not tokens:
            return ""

        try:
            # 토크나이저의 변환 함수 사용
            text = self.head_finder.tokenizer.convert_tokens_to_string(tokens)
        except Exception as e:
            logging.warning(f"토큰 변환 중 오류: {e}, 수동 변환 시도")
            # 수동 변환 (폴백)
            text = " ".join(tokens)
            # 특수 토큰 제거
            text = re.sub(r"<[^>]*>", "", text)
            text = re.sub(r"\[[^\]]*\]", "", text)

        # 후처리로 자연스러운 텍스트 만들기
        text = self._post_process_text(text)

        return text.strip()

    def _post_process_text(self, text: str) -> str:
        """텍스트 후처리"""
        # 기본 정리
        text = re.sub(r"\s+", " ", text)  # 중복 공백 제거
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)  # 구두점 앞 공백 제거
        text = re.sub(r"([,.!?;:])\s*([,.!?;:])", r"\1\2", text)  # 중복 구두점 정리

        # SentencePiece 토큰 정리
        text = re.sub(r"▁", " ", text)  # SentencePiece prefix 제거
        text = re.sub(r"Ġ", " ", text)  # GPT-2 space token 제거

        # 추가 정리
        text = re.sub(r"\s+", " ", text)  # 다시 한 번 공백 정리

        return text

    def compress_and_generate(
        self,
        text: str,
        compression_ratio: float = 0.3,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> Dict[str, Union[str, float, int]]:
        """
        프롬프트를 압축하고 같은 모델로 생성까지 수행 (개선된 버전)
        """
        # 1. 원본으로 생성
        original_result = self._generate_response(
            text, max_new_tokens, temperature, do_sample
        )

        # 2. 압축 후 생성
        compression_result = self.compress_prompt(text, compression_ratio)
        compressed_text = self.tokens_to_text(compression_result.compressed_tokens)

        compressed_result = self._generate_response(
            compressed_text, max_new_tokens, temperature, do_sample
        )

        # 결과 정리
        return {
            "original_text": text,
            "compressed_text": compressed_text,
            "original_response": original_result["response"],
            "compressed_response": compressed_result["response"],
            "compression_ratio": compression_result.compression_ratio,
            "original_length": len(text.split()),
            "compressed_length": len(compressed_text.split()),
            "tokens_kept": len(compression_result.selected_indices),
            "tokens_total": len(compression_result.original_tokens),
            "original_tokens_count": original_result["input_tokens"],
            "compressed_tokens_count": compressed_result["input_tokens"],
            "generation_time_original": original_result["generation_time"],
            "generation_time_compressed": compressed_result["generation_time"],
        }

    def _generate_response(
        self, text: str, max_new_tokens: int, temperature: float, do_sample: bool
    ) -> Dict:
        """응답 생성 (시간 측정 포함) - KoAlpaca 호환성 개선"""
        import time

        start_time = time.time()

        try:
            inputs = safe_tokenize(
                self.head_finder.tokenizer,
                text,
                model_name=self.model_name,
                model_config=self.head_finder.model_config,
                max_length=1024,
            )

            inputs = {k: v.to(self.head_finder.device) for k, v in inputs.items()}
            input_tokens = inputs["input_ids"].size(1)

            with torch.no_grad():
                # KoAlpaca 모델 특화 생성 설정
                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "do_sample": do_sample,
                    "pad_token_id": self.head_finder.tokenizer.eos_token_id,
                    "eos_token_id": self.head_finder.tokenizer.eos_token_id,
                    "repetition_penalty": 1.1,
                }

                # KoAlpaca 모델인 경우 추가 설정
                if "koalpaca" in self.model_name.lower():
                    generation_kwargs.update(
                        {
                            "top_p": 0.95,
                            "top_k": 50,
                            "no_repeat_ngram_size": 3,
                        }
                    )

                outputs = self.head_finder.model.generate(**inputs, **generation_kwargs)

            response = self.head_finder.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            generation_time = time.time() - start_time

            return {
                "response": response,
                "input_tokens": input_tokens,
                "generation_time": generation_time,
            }

        except Exception as e:
            logging.error(f"생성 중 오류: {e}")
            return {
                "response": f"[생성 오류: {str(e)}]",
                "input_tokens": 0,
                "generation_time": time.time() - start_time,
            }

    def get_compression_stats(self) -> Dict:
        """압축기 통계 정보"""
        model_info = self.head_finder.get_model_info_dict()

        return {
            "model_name": self.model_name,
            "model_info": model_info,
            "is_initialized": self.is_initialized,
            "num_evaluator_heads": (
                len(self.evaluator_heads) if self.evaluator_heads else 0
            ),
            "evaluator_heads": (
                [
                    {
                        "layer": head.layer,
                        "head": head.head,
                        "confidence": head.confidence_score,
                        "selectivity": head.selectivity_score,
                    }
                    for head in (self.evaluator_heads or [])
                ]
                if self.evaluator_heads
                else []
            ),
        }

    def benchmark_compression(
        self, test_texts: List[str], compression_ratios: List[float] = None
    ) -> Dict:
        """압축 성능 벤치마크"""
        if compression_ratios is None:
            compression_ratios = [0.2, 0.3, 0.5, 0.7]

        results = {}

        for ratio in compression_ratios:
            ratio_results = {
                "compression_ratio": ratio,
                "results": [],
                "avg_actual_ratio": 0,
                "avg_compression_time": 0,
            }

            total_actual_ratio = 0
            total_time = 0

            for text in test_texts:
                try:
                    import time

                    start_time = time.time()

                    result = self.compress_prompt(text, ratio)
                    compression_time = time.time() - start_time

                    ratio_results["results"].append(
                        {
                            "original_length": len(result.original_tokens),
                            "compressed_length": len(result.compressed_tokens),
                            "actual_ratio": result.compression_ratio,
                            "compression_time": compression_time,
                        }
                    )

                    total_actual_ratio += result.compression_ratio
                    total_time += compression_time

                except Exception as e:
                    logging.warning(f"벤치마크 중 오류: {e}")
                    continue

            if ratio_results["results"]:
                ratio_results["avg_actual_ratio"] = total_actual_ratio / len(
                    ratio_results["results"]
                )
                ratio_results["avg_compression_time"] = total_time / len(
                    ratio_results["results"]
                )

            results[str(ratio)] = ratio_results

        return results


# 편의 함수들
def create_compressor(
    model_name: Optional[str] = None, auto_initialize: bool = True
) -> EHPCCompressor:
    """간편한 압축기 생성 함수"""
    return EHPCCompressor(model_name, auto_initialize)


def quick_compress(
    text: str, ratio: float = 0.3, model_name: Optional[str] = None
) -> str:
    """빠른 압축 함수"""
    compressor = create_compressor(model_name, auto_initialize=True)
    result = compressor.compress_prompt(text, ratio)
    return compressor.tokens_to_text(result.compressed_tokens)


# 테스트 함수
def test_compression():
    """압축 기능 테스트"""
    test_text = """
    인공지능은 현대 기술의 핵심이며, 특히 자연어 처리 분야에서 혁신적인 발전을 보이고 있습니다.
    대규모 언어 모델들은 텍스트 이해와 생성에서 놀라운 성능을 보여주고 있으며,
    이는 다양한 실무 응용 분야에서 활용되고 있습니다. 특히 프롬프트 압축 기술은
    긴 텍스트 처리 시 발생하는 계산 비용과 메모리 사용량을 크게 줄일 수 있는 혁신적인 기술입니다.
    """

    try:
        print("🧪 압축 테스트 시작...")

        compressor = create_compressor(auto_initialize=True)
        print(f"✅ 압축기 생성 완료: {compressor.model_name}")

        result = compressor.compress_prompt(test_text, compression_ratio=0.3)
        compressed_text = compressor.tokens_to_text(result.compressed_tokens)

        print("📊 압축 결과:")
        print(f"   원본 토큰: {len(result.original_tokens)}")
        print(f"   압축 토큰: {len(result.compressed_tokens)}")
        print(f"   압축률: {result.compression_ratio:.1%}")
        print(f"   압축된 텍스트: {compressed_text[:100]}...")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    test_compression()
