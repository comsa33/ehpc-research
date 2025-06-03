import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class EvaluatorHeadInfo:
    """Evaluator Head 정보를 담는 데이터클래스"""

    layer: int
    head: int
    selectivity_score: float  # 선택성 점수 (낮을수록 집중적)
    confidence_score: float  # 신뢰도 점수


@dataclass
class CompressionResult:
    """압축 결과를 담는 데이터클래스"""

    original_tokens: List[str]
    compressed_tokens: List[str]
    token_scores: np.ndarray
    selected_indices: List[int]
    compression_ratio: float
    evaluator_heads: List[EvaluatorHeadInfo]


class EvaluatorHeadFinder:
    """논문의 핵심 - Evaluator Heads를 찾는 클래스"""

    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 디바이스 처리를 명확하게 수정
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
                torch_dtype=torch.float16,
                attn_implementation="eager",  # attention 구현 명시
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                output_attentions=True,
                torch_dtype=torch.float32,
                attn_implementation="eager",  # attention 구현 명시
            )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logging.info(f"Model loaded on {self.device}")

    def create_needle_haystack_data(
        self, num_samples: int = 20
    ) -> List[Dict[str, str]]:
        """
        논문에서 사용한 Needle-in-a-Haystack 테스트 데이터 생성
        긴 텍스트 중간에 특정 정보를 숨기고 찾을 수 있는지 테스트
        """
        haystack_templates = [
            "The weather was beautiful today with clear blue skies and gentle winds.",
            "Technology continues to advance at an unprecedented pace in various fields.",
            "Literature has always been a mirror reflecting human experiences and emotions.",
            "Mathematics provides the foundation for understanding patterns in nature.",
            "History teaches us valuable lessons about human civilization and progress.",
            "Music has the power to transcend cultural boundaries and unite people.",
            "Science explores the mysteries of the universe through systematic observation.",
            "Philosophy questions the fundamental nature of existence and knowledge.",
            "Art expresses human creativity and imagination in countless beautiful forms.",
            "Education empowers individuals to reach their full potential in life.",
        ]

        needles = [
            "The secret code is ALPHA-7842.",
            "The hidden treasure is buried under the old oak tree.",
            "The password for the system is 'BlueSky2024'.",
            "The meeting will be held at room 314 on Friday.",
            "The special ingredient is three tablespoons of vanilla extract.",
        ]

        test_data = []
        for i in range(num_samples):
            # 무작위로 haystack 문장들 선택
            haystack_sentences = np.random.choice(
                haystack_templates, size=np.random.randint(8, 15), replace=True
            )
            needle = np.random.choice(needles)

            # needle을 중간 위치에 삽입
            sentences = list(haystack_sentences)
            insert_pos = len(sentences) // 2
            sentences.insert(insert_pos, needle)

            full_text = " ".join(sentences)
            test_data.append(
                {
                    "text": full_text,
                    "needle": needle,
                    "answer": needle.split()[-1],  # 마지막 단어를 답으로 사용
                }
            )

        return test_data

    def evaluate_head_performance(
        self, layer_idx: int, head_idx: int, test_data: List[Dict[str, str]]
    ) -> float:
        """
        특정 헤드가 Needle-in-a-Haystack 작업에서 얼마나 잘 수행하는지 평가
        """
        correct_predictions = 0

        for data in test_data:
            text = data["text"]
            needle = data["needle"]

            # 토큰화 및 디바이스로 이동
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            with torch.no_grad():
                outputs = self.model(**inputs)
                attentions = outputs.attentions

            if layer_idx < len(attentions):
                # 특정 헤드의 attention 가져오기
                attention = attentions[layer_idx][0, head_idx]  # [seq_len, seq_len]

                # needle 토큰들의 인덱스 찾기
                needle_tokens = self.tokenizer.tokenize(needle)
                needle_indices = []

                for i, token in enumerate(tokens):
                    if any(needle_token in token for needle_token in needle_tokens):
                        needle_indices.append(i)

                if needle_indices:
                    # 마지막 토큰이 needle 토큰들에 높은 attention을 주는지 확인
                    last_token_attention = attention[-1]  # 마지막 토큰의 attention
                    needle_attention_sum = sum(
                        last_token_attention[idx] for idx in needle_indices
                    )

                    # 전체 attention 대비 needle에 대한 attention 비율
                    if (
                        needle_attention_sum / last_token_attention.sum() > 0.1
                    ):  # 임계값
                        correct_predictions += 1

        return correct_predictions / len(test_data) if test_data else 0.0

    def find_evaluator_heads(
        self, max_layers: int = 3, heads_per_layer: int = 2
    ) -> List[EvaluatorHeadInfo]:
        """
        Evaluator Heads를 찾는 메인 함수
        1. Needle-in-a-Haystack 테스트 생성
        2. 각 헤드의 성능 평가
        3. 최고 성능 헤드들 선택
        """
        logging.info("Creating Needle-in-a-Haystack test data...")
        test_data = self.create_needle_haystack_data(num_samples=50)

        logging.info("Evaluating attention heads...")
        head_scores = []

        # 모델의 첫 몇 레이어만 검사 (논문에서 초기 레이어가 더 효과적)
        # 수정: len() 제거하고 직접 값 사용
        num_layers = min(
            max_layers, self.model.config.to_dict().get("num_hidden_layers", 12)
        )
        num_heads = self.model.config.to_dict().get("num_attention_heads", 12)

        logging.info(f"Checking {num_layers} layers with {num_heads} heads each...")

        for layer_idx in range(num_layers):
            for head_idx in range(num_heads):
                # 각 헤드의 성능 점수 계산
                performance_score = self.evaluate_head_performance(
                    layer_idx, head_idx, test_data
                )

                # 추가적인 selectivity 점수 계산 (attention entropy 기반)
                selectivity_score = self._calculate_head_selectivity(
                    layer_idx, head_idx, test_data
                )

                head_info = EvaluatorHeadInfo(
                    layer=layer_idx,
                    head=head_idx,
                    selectivity_score=selectivity_score,
                    confidence_score=performance_score,
                )
                head_scores.append(head_info)

        # 성능 순으로 정렬하고 각 레이어에서 최고 헤드들 선택
        head_scores.sort(
            key=lambda x: (x.confidence_score + (1.0 - x.selectivity_score)),
            reverse=True,
        )

        selected_heads = []
        layer_counts = {}

        for head_info in head_scores:
            layer = head_info.layer
            if layer_counts.get(layer, 0) < heads_per_layer:
                selected_heads.append(head_info)
                layer_counts[layer] = layer_counts.get(layer, 0) + 1

        logging.info(f"Selected {len(selected_heads)} evaluator heads:")
        for head in selected_heads:
            logging.info(
                f"  Layer {head.layer}, Head {head.head}: confidence={head.confidence_score:.3f}, selectivity={head.selectivity_score:.3f}"
            )

        return selected_heads

    def _calculate_head_selectivity(
        self, layer_idx: int, head_idx: int, test_data: List[Dict[str, str]]
    ) -> float:
        """헤드의 선택성 점수 계산 (attention이 얼마나 집중되는지)"""
        total_entropy = 0
        valid_samples = 0

        for data in test_data[:10]:  # 샘플링하여 계산 속도 향상
            text = data["text"]
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            # 디바이스로 이동
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                attentions = outputs.attentions

            if layer_idx < len(attentions):
                attention = attentions[layer_idx][0, head_idx]

                # 각 쿼리 위치에서 attention distribution의 entropy 계산
                attention_probs = torch.softmax(attention, dim=-1)
                entropy = -torch.sum(
                    attention_probs * torch.log(attention_probs + 1e-12), dim=-1
                )
                total_entropy += entropy.mean().item()
                valid_samples += 1

        return total_entropy / valid_samples if valid_samples > 0 else 1.0
