import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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


class ModelConfig:
    """모델별 설정 정보"""

    SUPPORTED_MODELS = {
        # 추천 모델들 (성능 순)
        "google/gemma-2-2b": {
            "params": "2B",
            "min_memory_gb": 4,
            "quantize": True,
            "context_length": 8192,
            "pros": ["빠른 속도", "최신 아키텍처", "안정적"],
            "cons": ["작은 모델 크기"],
            "korean_support": 3,  # 1-5 점수
        },
        "Qwen/Qwen2.5-3B-Instruct": {
            "params": "3B",
            "min_memory_gb": 6,
            "quantize": True,
            "context_length": 32768,
            "pros": ["한국어 지원 우수", "효율적", "최신 기술"],
            "cons": ["중국 회사 모델"],
            "korean_support": 5,
        },
        "meta-llama/Llama-3.2-3B-Instruct": {
            "params": "3B",
            "min_memory_gb": 8,
            "quantize": True,
            "context_length": 131072,
            "pros": ["최신 아키텍처", "긴 컨텍스트", "Meta 공식"],
            "cons": ["라이센스 제약"],
            "korean_support": 3,
        },
        "microsoft/Phi-3.5-mini-instruct": {
            "params": "3.8B",
            "min_memory_gb": 6,
            "quantize": True,
            "context_length": 128000,
            "pros": ["Microsoft 최신", "효율적", "긴 컨텍스트"],
            "cons": ["상대적으로 새로움"],
            "korean_support": 3,
        },
        # 한국어 특화 모델들
        "beomi/KoAlpaca-Polyglot-5.8B": {
            "params": "5.8B",
            "min_memory_gb": 12,
            "quantize": True,
            "context_length": 2048,
            "pros": ["한국어 최고 성능", "학술 연구용"],
            "cons": ["큰 메모리 사용"],
            "korean_support": 5,
        },
        # 기존 호환성 모델
        "microsoft/DialoGPT-medium": {
            "params": "354M",
            "min_memory_gb": 2,
            "quantize": False,
            "context_length": 1024,
            "pros": ["가벼움", "기존 호환성"],
            "cons": ["구식 아키텍처", "성능 제한"],
            "korean_support": 1,
        },
    }

    @classmethod
    def get_recommended_model(cls) -> str:
        """하드웨어 환경에 맞는 모델 추천"""
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logging.info(f"🎮 GPU 메모리: {gpu_memory_gb:.1f}GB")

            if gpu_memory_gb >= 16:
                return "beomi/KoAlpaca-Polyglot-5.8B"  # 한국어 최고 성능
            elif gpu_memory_gb >= 12:
                return "meta-llama/Llama-3.2-3B-Instruct"  # 고성능
            elif gpu_memory_gb >= 8:
                return "Qwen/Qwen2.5-3B-Instruct"  # 균형
            else:
                return "google/gemma-2-2b"  # 경량

        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # Mac M1/M2/M3
            import psutil

            total_memory_gb = psutil.virtual_memory().total / (1024**3)
            logging.info(f"🍎 Mac 메모리: {total_memory_gb:.1f}GB")

            if total_memory_gb >= 32:
                return "Qwen/Qwen2.5-3B-Instruct"
            else:
                return "google/gemma-2-2b"
        else:
            # CPU 환경
            logging.info("💻 CPU 환경 감지")
            return "microsoft/DialoGPT-medium"


class EvaluatorHeadFinder:
    """논문의 핵심 - Evaluator Heads를 찾는 클래스 (업그레이드 버전)"""

    def __init__(self, model_name: Optional[str] = None):
        # 모델 자동 선택
        if model_name is None:
            model_name = ModelConfig.get_recommended_model()
            logging.info(f"🎯 자동 선택된 모델: {model_name}")

        self.model_name = model_name
        self.model_config = ModelConfig.SUPPORTED_MODELS.get(model_name, {})
        self.device = self._get_optimal_device()

        logging.info(f"📚 모델 로딩 시작: {model_name}")
        logging.info(f"⚙️ 설정: {self.model_config}")

        # 토크나이저 로드
        self._load_tokenizer()

        # 모델 로드
        self._load_model()

        self.model.eval()
        logging.info(f"✅ 모델 로딩 완료: {self._get_model_stats()}")

    def _get_hf_token(self) -> Optional[str]:
        """Hugging Face 토큰 가져오기"""

        # 1. 환경 변수에서 토큰 찾기
        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        if token:
            logging.info("🔑 환경 변수에서 HF 토큰 발견")
            return token

        # 2. .env 파일에서 토큰 찾기
        try:
            from pathlib import Path

            env_file = Path(".env")
            if env_file.exists():
                with open(env_file, "r") as f:
                    for line in f:
                        if line.startswith("HUGGINGFACE_TOKEN=") or line.startswith(
                            "HF_TOKEN="
                        ):
                            token = line.split("=", 1)[1].strip().strip("\"'")
                            logging.info("🔑 .env 파일에서 HF 토큰 발견")
                            return token
        except Exception as e:
            logging.debug(f".env 파일 읽기 실패: {e}")

        # 3. Hugging Face CLI 캐시에서 토큰 찾기
        try:
            from huggingface_hub import HfFolder

            cached_token = HfFolder.get_token()
            if cached_token:
                logging.info("🔑 HF CLI 캐시에서 토큰 발견")
                return cached_token
        except Exception as e:
            logging.debug(f"HF CLI 토큰 가져오기 실패: {e}")

        # 4. 토큰이 없으면 로그인 안내
        logging.warning("⚠️ Hugging Face 토큰을 찾을 수 없습니다.")
        logging.info("다음 중 하나의 방법으로 로그인하세요:")
        logging.info("1. 환경 변수: export HUGGINGFACE_TOKEN='your_token'")
        logging.info("2. .env 파일: echo 'HUGGINGFACE_TOKEN=your_token' > .env")
        logging.info("3. CLI 로그인: huggingface-cli login")

        return None

    def _handle_gated_model_fallback(self, original_model: str) -> str:
        """Gated 모델 접근 실패 시 폴백 모델 선택"""
        fallback_models = {
            "google/gemma-2-2b": "microsoft/DialoGPT-medium",
            "google/gemma-2-2b-it": "microsoft/DialoGPT-medium",
            "meta-llama/Llama-3.2-3B-Instruct": "microsoft/Phi-3.5-mini-instruct",
            "meta-llama/Llama-3.2-8B-Instruct": "Qwen/Qwen2.5-3B-Instruct",
            "beomi/KoAlpaca-Polyglot-5.8B": "Qwen/Qwen2.5-3B-Instruct",
        }

        fallback = fallback_models.get(original_model, "microsoft/DialoGPT-medium")
        logging.info(f"🔄 Gated 모델 {original_model} → 폴백 모델 {fallback}")
        return fallback

    def _get_optimal_device(self) -> str:
        """최적 디바이스 자동 선택"""
        if torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            gpu_name = torch.cuda.get_device_name(0)
            logging.info(f"🎮 CUDA 디바이스: {gpu_name}")
            return device
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logging.info("🍎 Apple MPS 디바이스 사용")
            return "mps"
        else:
            logging.info("💻 CPU 디바이스 사용")
            return "cpu"

    def _load_tokenizer(self):
        """토크나이저 로드"""
        try:
            # Hugging Face 인증 처리
            token = self._get_hf_token()

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True,
                token=token,  # 인증 토큰 추가
            )

            # 패드 토큰 설정
            if self.tokenizer.pad_token is None:
                if self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

            logging.info(f"📝 토크나이저 로드 완료: vocab_size={len(self.tokenizer)}")

        except Exception as e:
            logging.error(f"❌ 토크나이저 로딩 실패: {e}")
            raise

    def _load_model(self):
        """최적화된 모델 로딩"""
        try:
            # Hugging Face 인증 처리
            token = self._get_hf_token()

            model_kwargs = {
                "output_attentions": True,
                "trust_remote_code": True,
                "attn_implementation": "eager",  # 안정성 우선
                "token": token,  # 인증 토큰 추가
            }

            # 양자화 설정 (CUDA에서만)
            if (
                self.model_config.get("quantize", False)
                and "cuda" in self.device
                and self._is_quantization_available()
            ):

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["torch_dtype"] = torch.float16
                logging.info("🔧 8bit 양자화 적용")

            elif "mps" in self.device:
                model_kwargs["torch_dtype"] = torch.float16
                logging.info("🍎 MPS용 float16 설정")
            else:
                model_kwargs["torch_dtype"] = torch.float32
                logging.info("💻 CPU용 float32 설정")

            # 모델 로드
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )

            # 디바이스로 이동 (양자화되지 않은 경우만)
            if "quantization_config" not in model_kwargs:
                self.model = self.model.to(self.device)

        except Exception as e:
            logging.warning(f"⚠️ 모델 로딩 실패, 폴백 시도: {e}")

            # Gated 모델 접근 오류인지 확인
            if "gated repo" in str(e).lower() or "access to model" in str(e).lower():
                fallback_model = self._handle_gated_model_fallback(self.model_name)
                logging.info(f"🔄 폴백 모델로 재시도: {fallback_model}")

                # 폴백 모델로 재귀 호출
                original_model = self.model_name
                self.model_name = fallback_model
                self.model_config = ModelConfig.SUPPORTED_MODELS.get(fallback_model, {})

                try:
                    self._load_tokenizer()  # 토크나이저도 다시 로드
                    self._load_model()  # 모델 다시 로드
                    logging.info(f"✅ 폴백 모델 {fallback_model} 로드 성공")
                    return
                except Exception as fallback_error:
                    logging.error(f"❌ 폴백 모델도 실패: {fallback_error}")
                    # 원래 모델명 복구 후 일반 폴백으로 넘어감
                    self.model_name = original_model

            self._load_fallback_model()

    def _is_quantization_available(self) -> bool:
        """양자화 라이브러리 사용 가능 여부 확인"""
        try:
            import bitsandbytes

            return True
        except ImportError:
            logging.warning("⚠️ bitsandbytes 라이브러리 없음, 양자화 비활성화")
            return False

    def _load_fallback_model(self):
        """폴백 모델 로딩"""
        fallback_models = ["google/gemma-2-2b", "microsoft/DialoGPT-medium", "gpt2"]

        for fallback in fallback_models:
            try:
                logging.info(f"🔄 폴백 모델 시도: {fallback}")

                self.model = AutoModelForCausalLM.from_pretrained(
                    fallback,
                    output_attentions=True,
                    torch_dtype=torch.float32,
                    attn_implementation="eager",
                )

                # 토크나이저도 다시 로드
                self.tokenizer = AutoTokenizer.from_pretrained(fallback)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = self.model.to(self.device)
                self.model_name = fallback
                self.model_config = ModelConfig.SUPPORTED_MODELS.get(fallback, {})

                logging.info(f"✅ 폴백 모델 로딩 성공: {fallback}")
                return

            except Exception as e:
                logging.warning(f"❌ 폴백 모델 {fallback} 실패: {e}")
                continue

        raise RuntimeError("❌ 모든 모델 로딩 시도 실패")

    def _get_model_stats(self) -> str:
        """모델 통계 정보"""
        try:
            param_count = sum(p.numel() for p in self.model.parameters()) / 1e6
            device_info = str(next(self.model.parameters()).device)
            dtype_info = str(next(self.model.parameters()).dtype)

            memory_usage = ""
            if "cuda" in device_info:
                memory_gb = torch.cuda.memory_allocated() / (1024**3)
                memory_usage = f", {memory_gb:.1f}GB 사용"

            return f"{param_count:.1f}M params on {device_info} ({dtype_info}){memory_usage}"
        except:
            return "통계 정보 없음"

    def create_needle_haystack_data(
        self, num_samples: int = 20
    ) -> List[Dict[str, str]]:
        """
        논문에서 사용한 Needle-in-a-Haystack 테스트 데이터 생성 (한국어 추가)
        """
        # 영어 templates
        haystack_templates_en = [
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

        # 한국어 templates (한국어 지원 모델용)
        haystack_templates_ko = [
            "오늘 날씨는 맑은 하늘과 부드러운 바람으로 매우 아름다웠습니다.",
            "기술은 다양한 분야에서 전례 없는 속도로 계속 발전하고 있습니다.",
            "문학은 항상 인간의 경험과 감정을 반영하는 거울 역할을 해왔습니다.",
            "수학은 자연의 패턴을 이해하는 기초를 제공합니다.",
            "역사는 인간 문명과 발전에 대한 귀중한 교훈을 가르쳐 줍니다.",
            "음악은 문화적 경계를 넘나들며 사람들을 하나로 묶는 힘이 있습니다.",
            "과학은 체계적인 관찰을 통해 우주의 신비를 탐구합니다.",
            "철학은 존재와 지식의 근본적 본질에 대해 질문합니다.",
        ]

        needles_en = [
            "The secret code is ALPHA-7842.",
            "The hidden treasure is buried under the old oak tree.",
            "The password for the system is 'BlueSky2024'.",
            "The meeting will be held at room 314 on Friday.",
            "The special ingredient is three tablespoons of vanilla extract.",
        ]

        needles_ko = [
            "비밀 코드는 ALPHA-7842입니다.",
            "숨겨진 보물은 오래된 참나무 아래에 묻혀 있습니다.",
            "시스템 비밀번호는 'BlueSky2024'입니다.",
            "회의는 금요일 314호실에서 열릴 예정입니다.",
            "특별한 재료는 바닐라 추출물 3큰술입니다.",
        ]

        # 한국어 지원 점수에 따라 언어 선택
        korean_support = self.model_config.get("korean_support", 1)
        use_korean = korean_support >= 4

        haystack_templates = (
            haystack_templates_ko if use_korean else haystack_templates_en
        )
        needles = needles_ko if use_korean else needles_en

        logging.info(f"🌐 테스트 언어: {'한국어' if use_korean else '영어'}")

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
                    "language": "korean" if use_korean else "english",
                }
            )

        return test_data

    def evaluate_head_performance(
        self, layer_idx: int, head_idx: int, test_data: List[Dict[str, str]]
    ) -> float:
        """특정 헤드가 Needle-in-a-Haystack 작업에서 얼마나 잘 수행하는지 평가 (안정성 강화)"""
        correct_predictions = 0
        valid_samples = 0

        for data in test_data:
            try:
                text = data["text"]
                needle = data["needle"]

                # 토큰화 및 길이 제한
                max_length = min(self.model_config.get("context_length", 2048), 1024)
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_length,
                    padding=False,
                )

                # 디바이스로 이동
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # 빈 입력 체크
                if inputs["input_ids"].size(1) == 0:
                    continue

                tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    attentions = outputs.attentions

                # 안전한 인덱스 체크
                if layer_idx >= len(attentions) or head_idx >= attentions[
                    layer_idx
                ].size(1):
                    continue

                attention = attentions[layer_idx][0, head_idx]  # [seq_len, seq_len]

                if attention.size(0) == 0 or attention.size(1) == 0:
                    continue

                # needle 토큰들의 인덱스 찾기 (개선된 방법)
                needle_tokens = self.tokenizer.tokenize(needle)
                needle_indices = []

                # 더 정확한 needle 매칭
                needle_text_lower = needle.lower()
                for i, token in enumerate(tokens):
                    if i < attention.size(0):
                        token_text = (
                            self.tokenizer.convert_tokens_to_string([token])
                            .lower()
                            .strip()
                        )
                        if any(
                            needle_word in token_text
                            for needle_word in needle_text_lower.split()
                        ):
                            needle_indices.append(i)

                if needle_indices and len(tokens) > 0:
                    # 마지막 토큰의 attention 패턴 분석
                    last_token_idx = min(len(tokens) - 1, attention.size(0) - 1)
                    if last_token_idx >= 0:
                        last_token_attention = attention[last_token_idx]

                        # 유효한 needle 인덱스들만 사용
                        valid_needle_indices = [
                            idx
                            for idx in needle_indices
                            if idx < last_token_attention.size(0)
                        ]

                        if valid_needle_indices:
                            needle_attention_sum = sum(
                                last_token_attention[idx]
                                for idx in valid_needle_indices
                            )

                            total_attention = last_token_attention.sum()
                            if total_attention > 0:
                                attention_ratio = needle_attention_sum / total_attention
                                # 임계값을 모델에 따라 조정
                                threshold = (
                                    0.15
                                    if self.model_config.get("params", "").endswith("B")
                                    else 0.1
                                )

                                if attention_ratio > threshold:
                                    correct_predictions += 1

                valid_samples += 1

            except (RuntimeError, IndexError, KeyError) as e:
                logging.debug(f"샘플 처리 중 오류 (스킵): {e}")
                continue

        return correct_predictions / valid_samples if valid_samples > 0 else 0.0

    def find_evaluator_heads(
        self, max_layers: int = 3, heads_per_layer: int = 2
    ) -> List[EvaluatorHeadInfo]:
        """Evaluator Heads를 찾는 메인 함수 (개선된 버전)"""
        logging.info("🔍 Needle-in-a-Haystack 테스트 데이터 생성 중...")
        test_data = self.create_needle_haystack_data(num_samples=30)  # 샘플 수 증가

        logging.info("🧠 Attention heads 평가 중...")
        head_scores = []

        # 모델 설정 가져오기
        try:
            config = self.model.config
            num_layers = min(max_layers, getattr(config, "num_hidden_layers", 12))
            num_heads = getattr(config, "num_attention_heads", 12)

            logging.info(
                f"📊 검사 범위: {num_layers}개 레이어, 레이어당 {num_heads}개 헤드"
            )

        except Exception as e:
            logging.warning(f"모델 설정 가져오기 실패: {e}. 기본값 사용.")
            num_layers = min(max_layers, 6)
            num_heads = 8

        # 진행 상황 추적
        total_heads = num_layers * num_heads
        current_head = 0

        for layer_idx in range(num_layers):
            layer_heads = []

            for head_idx in range(num_heads):
                current_head += 1
                progress = (current_head / total_heads) * 100

                if current_head % 5 == 0:  # 5개마다 로그
                    logging.info(
                        f"진행률: {progress:.1f}% ({current_head}/{total_heads})"
                    )

                try:
                    # 성능 점수 계산
                    performance_score = self.evaluate_head_performance(
                        layer_idx, head_idx, test_data
                    )

                    # 선택성 점수 계산
                    selectivity_score = self._calculate_head_selectivity(
                        layer_idx, head_idx, test_data[:5]  # 샘플링으로 속도 향상
                    )

                    head_info = EvaluatorHeadInfo(
                        layer=layer_idx,
                        head=head_idx,
                        selectivity_score=selectivity_score,
                        confidence_score=performance_score,
                    )

                    layer_heads.append(head_info)

                except Exception as e:
                    logging.debug(f"헤드 {layer_idx}-{head_idx} 평가 중 오류: {e}")
                    continue

            # 레이어별 상위 헤드들만 선택
            if layer_heads:
                layer_heads.sort(
                    key=lambda x: x.confidence_score + (1.0 - x.selectivity_score),
                    reverse=True,
                )
                head_scores.extend(layer_heads[:heads_per_layer])

        if not head_scores:
            # 백업: 기본 헤드 선택
            logging.warning("⚠️ 유효한 헤드를 찾지 못함, 기본 헤드 사용")
            default_heads = [
                EvaluatorHeadInfo(
                    layer=0, head=0, selectivity_score=0.5, confidence_score=0.1
                ),
                EvaluatorHeadInfo(
                    layer=1, head=0, selectivity_score=0.5, confidence_score=0.1
                ),
            ]
            return default_heads

        # 최종 선택: 전체 중 상위 헤드들
        head_scores.sort(
            key=lambda x: x.confidence_score + (1.0 - x.selectivity_score), reverse=True
        )

        # 최대 max_layers * heads_per_layer 개 선택
        max_selected = max_layers * heads_per_layer
        selected_heads = head_scores[:max_selected]

        logging.info(f"✅ {len(selected_heads)}개 Evaluator Heads 선택 완료:")
        for head in selected_heads:
            logging.info(
                f"  📍 Layer {head.layer}, Head {head.head}: "
                f"신뢰도={head.confidence_score:.3f}, 선택성={head.selectivity_score:.3f}"
            )

        return selected_heads

    def _calculate_head_selectivity(
        self, layer_idx: int, head_idx: int, test_data: List[Dict[str, str]]
    ) -> float:
        """헤드의 선택성 점수 계산 (안정성 강화)"""
        total_entropy = 0
        valid_samples = 0

        for data in test_data:
            try:
                text = data["text"]
                max_length = min(self.model_config.get("context_length", 2048), 512)
                inputs = self.tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                if inputs["input_ids"].size(1) == 0:
                    continue

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    attentions = outputs.attentions

                if layer_idx < len(attentions) and head_idx < attentions[
                    layer_idx
                ].size(1):

                    attention = attentions[layer_idx][0, head_idx]

                    if attention.size(0) > 0 and attention.size(1) > 0:
                        # Attention을 확률 분포로 변환
                        attention_probs = torch.softmax(attention, dim=-1)

                        # NaN/무한값 체크
                        if torch.isfinite(attention_probs).all():
                            # 엔트로피 계산 (클램핑으로 안정성 확보)
                            log_probs = torch.log(
                                torch.clamp(attention_probs, min=1e-12)
                            )
                            entropy = -torch.sum(attention_probs * log_probs, dim=-1)

                            if torch.isfinite(entropy).all():
                                total_entropy += entropy.mean().item()
                                valid_samples += 1

            except (RuntimeError, IndexError, KeyError) as e:
                logging.debug(f"선택성 계산 중 오류 (스킵): {e}")
                continue

        return total_entropy / valid_samples if valid_samples > 0 else 1.0

    def get_model_info_dict(self) -> Dict:
        """모델 정보 딕셔너리 반환"""
        try:
            config = self.model.config
            return {
                "model_name": self.model_name,
                "parameters": self.model_config.get("params", "Unknown"),
                "device": str(self.device),
                "num_layers": getattr(config, "num_hidden_layers", "Unknown"),
                "num_heads": getattr(config, "num_attention_heads", "Unknown"),
                "context_length": self.model_config.get("context_length", "Unknown"),
                "korean_support": self.model_config.get("korean_support", 1),
                "pros": self.model_config.get("pros", []),
                "quantized": self.model_config.get("quantize", False),
                "memory_usage": self._get_memory_usage(),
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_memory_usage(self) -> str:
        """메모리 사용량 반환"""
        try:
            if "cuda" in str(self.device):
                allocated = torch.cuda.memory_allocated() / (1024**3)
                cached = torch.cuda.memory_reserved() / (1024**3)
                return f"GPU: {allocated:.1f}GB allocated, {cached:.1f}GB cached"
            else:
                return "Non-CUDA device"
        except:
            return "Unknown"


# 편의 함수들
def get_recommended_model() -> str:
    """사용자 환경에 맞는 추천 모델 반환"""
    return ModelConfig.get_recommended_model()


def list_supported_models() -> List[str]:
    """지원되는 모델 리스트 반환"""
    return list(ModelConfig.SUPPORTED_MODELS.keys())


def get_model_info(model_name: str) -> Dict:
    """특정 모델의 정보 반환"""
    return ModelConfig.SUPPORTED_MODELS.get(model_name, {})


# 테스트 함수
def test_model_loading(model_name: Optional[str] = None):
    """모델 로딩 테스트"""
    try:
        if model_name is None:
            model_name = get_recommended_model()

        print(f"🧪 테스트 모델: {model_name}")

        finder = EvaluatorHeadFinder(model_name)
        print("✅ 모델 로딩 성공!")

        info = finder.get_model_info_dict()
        print(f"📊 모델 정보: {info}")

        # 간단한 헤드 찾기 테스트
        print("🔍 Evaluator Heads 찾기 테스트...")
        heads = finder.find_evaluator_heads(max_layers=2, heads_per_layer=1)
        print(f"✅ {len(heads)}개 헤드 발견!")

        return True

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        return False


if __name__ == "__main__":
    # 지원 모델 리스트 출력
    print("🎯 지원되는 모델들:")
    for model in list_supported_models():
        info = get_model_info(model)
        print(
            f"  - {model}: {info.get('params', 'Unknown')} ({info.get('korean_support', 1)}/5 한국어)"
        )

    print(f"\n🚀 추천 모델: {get_recommended_model()}")

    # 테스트 실행
    print("\n🧪 모델 로딩 테스트 시작...")
    test_model_loading()
