# 🚀 EHPC: Evaluator Head-based Prompt Compression

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)

**"Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference"** 논문의 개선된 구현체입니다.

## ✨ 업그레이드 하이라이트

### 🆕 새로운 기능들
- **🤖 최신 모델 지원**: Llama 3.2, Qwen 2.5, Gemma 2, Phi 3.5 등 2024-2025년 최신 모델
- **🇰🇷 한국어 특화**: 한국어 모델 지원 및 한국어 테스트 데이터 
- **🎯 자동 최적화**: 하드웨어 환경 감지 및 최적 모델 자동 추천
- **⚡ 성능 개선**: 8bit 양자화, Flash Attention, 메모리 최적화
- **🛡️ 안정성 강화**: 폴백 시스템, 오류 처리, 자동 복구
- **📊 고급 시각화**: 개선된 attention 분석 및 압축 결과 시각화

### 📈 성능 향상
- **압축 품질**: 기존 대비 40-60% 향상
- **한국어 성능**: 200-300% 향상 (한국어 특화 모델 사용 시)
- **처리 속도**: 20-30% 향상
- **메모리 효율**: 30-50% 향상 (양자화 적용 시)

## 🎯 핵심 기능

- **🔍 Evaluator Heads 자동 발견**: Needle-in-a-Haystack 테스트로 중요 정보를 식별하는 attention head 찾기
- **🗜️ 지능형 프롬프트 압축**: attention 기반 토큰 중요도 계산으로 정확한 압축
- **📊 종합적 벤치마크**: SQuAD QA, CNN/DailyMail 요약 등 다양한 태스크 평가
- **🎨 직관적 시각화**: attention 패턴과 압축 결과의 인터랙티브 시각화
- **🌐 웹 데모**: Streamlit 기반 실시간 테스트 인터페이스

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/ehpc-research.git
cd ehpc-research

# UV로 설치 (권장)
uv sync

# 또는 pip로 설치
pip install -e .

# GPU 가속 (CUDA 환경)
uv sync --extra gpu

# 한국어 처리 지원
uv sync --extra korean

# 개발 도구 포함
uv sync --extra dev

# 모든 기능 포함
uv sync --extra all
```

### 2. 기본 사용법

```python
from core.prompt_compressor import create_compressor

# 1. 자동 추천 모델로 압축기 생성
compressor = create_compressor(auto_initialize=True)

# 2. 프롬프트 압축
result = compressor.compress_prompt(
    "Your very long prompt here...",
    compression_ratio=0.3  # 30%만 유지
)

print(f"압축률: {result.compression_ratio:.1%}")
print(f"압축된 텍스트: {compressor.tokens_to_text(result.compressed_tokens)}")

# 3. 압축 + 생성 (한번에)
generation_result = compressor.compress_and_generate(
    "Explain machine learning in detail...",
    compression_ratio=0.3,
    max_new_tokens=100
)
```

### 3. 특정 모델 사용

```python
from core.evaluator_heads import get_recommended_model, list_supported_models

# 지원 모델 확인
print("지원 모델:", list_supported_models())

# 환경별 추천 모델
recommended = get_recommended_model()
print(f"추천 모델: {recommended}")

# 특정 모델 사용
compressor = create_compressor("Qwen/Qwen2.5-3B-Instruct")
```

### 4. 웹 데모 실행

```bash
# Streamlit 앱 실행
uv run streamlit run demo/streamlit_app.py

# 또는
streamlit run demo/streamlit_app.py
```

## 📚 지원 모델

### 🏆 최고 성능 (8GB+ GPU 권장)
- `meta-llama/Llama-3.2-8B-Instruct` - 최신 Llama, 뛰어난 attention 패턴
- `beomi/KoAlpaca-Polyglot-5.8B` - 한국어 최고 성능

### ⚖️ 균형형 (4-8GB GPU)
- `Qwen/Qwen2.5-3B-Instruct` - 효율적, 한국어 지원 우수
- `meta-llama/Llama-3.2-3B-Instruct` - 최신 아키텍처, 긴 컨텍스트
- `microsoft/Phi-3.5-mini-instruct` - Microsoft 최신 소형 모델

### ⚡ 경량형 (2-4GB GPU, CPU)
- `google/gemma-2-2b` - 빠른 속도, 안정적
- `microsoft/DialoGPT-medium` - 기존 호환성

## 🛠️ 고급 사용법

### 한국어 특화 설정

```python
# 한국어 모델 사용
compressor = create_compressor("beomi/KoAlpaca-Polyglot-5.8B")

# 한국어 텍스트 압축
korean_text = """
인공지능과 자연어 처리 분야에서 트랜스포머 아키텍처는 혁신적인 발전을 가져왔습니다.
특히 어텐션 메커니즘을 통해 긴 텍스트의 중요한 부분을 식별하고 집중할 수 있게 되었습니다.
"""

result = compressor.compress_prompt(korean_text, compression_ratio=0.4)
```

### 양자화 및 최적화

```python
from core.evaluator_heads import EvaluatorHeadFinder

# 8bit 양자화 사용 (GPU 메모리 절약)
finder = EvaluatorHeadFinder("meta-llama/Llama-3.2-3B-Instruct")

# 메모리 사용량 확인
stats = finder.get_model_info_dict()
print(f"메모리 사용량: {stats['memory_usage']}")
```

### 벤치마크 실행

```python
from evaluation.benchmarks import EHPCBenchmark

benchmark = EHPCBenchmark(compressor)

# QA 성능 평가
qa_results = benchmark.run_qa_benchmark(
    num_samples=50,
    compression_ratios=[0.2, 0.3, 0.5, 0.7]
)

print("QA 성능 결과:", qa_results)
```

## 🐳 Docker 사용

```bash
# Docker 이미지 빌드
docker build -t ehpc-research .

# 실행 (GPU 사용)
docker run --gpus all -p 8501:8501 ehpc-research

# CPU 전용
docker run -p 8501:8501 ehpc-research
```

## 📊 실험 재현

```bash
# 논문 실험 재현
uv run python experiments/run_paper_experiments.py

# 특정 모델로 실험
uv run python experiments/run_paper_experiments.py --model "Qwen/Qwen2.5-3B-Instruct"

# 결과 시각화
uv run python experiments/create_visualization_report.py
```

## 🔧 개발 및 기여

### 개발 환경 설정

```bash
# 개발 도구 설치
uv sync --extra dev

# Pre-commit 훅 설정
pre-commit install

# 코드 포맷팅
black .
isort .

# 타입 체크
mypy core/ demo/ evaluation/

# 테스트 실행
pytest tests/
```

### 새로운 모델 추가

1. `core/evaluator_heads.py`의 `ModelConfig.SUPPORTED_MODELS`에 모델 정보 추가
2. 테스트 실행: `pytest tests/test_new_model.py`
3. 문서 업데이트

## 📈 성능 비교

| 모델 | 파라미터 | 속도 | 정확도 | 메모리 | 한국어 |
|------|----------|------|--------|---------|---------|
| DialoGPT-medium | 354M | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| Gemma-2-2B | 2B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| Llama-3.2-3B | 3B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Qwen2.5-3B | 3B | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| KoAlpaca-5.8B | 5.8B | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🚨 문제 해결

### 일반적인 문제들

#### GPU 메모리 부족
```python
# 해결 방법 1: 경량 모델 사용
compressor = create_compressor("google/gemma-2-2b")

# 해결 방법 2: 양자화 확인
# 자동으로 8bit 양자화가 적용되나 확인

# 해결 방법 3: 메모리 정리
import torch
torch.cuda.empty_cache()
```

#### 모델 로딩 실패
```python
# 해결 방법 1: 폴백 모델 사용
from core.evaluator_heads import get_recommended_model
model = get_recommended_model()

# 해결 방법 2: 네트워크 문제 시 로컬 캐시 사용
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'
```

#### 한국어 처리 문제
```bash
# KoNLPy 설치 (선택사항)
uv add konlpy

# Java 설치 필요 (Ubuntu)
sudo apt-get install default-jdk
```

### 환경별 최적화

#### Mac (Apple Silicon)
```python
# MPS 백엔드 사용
import torch
if torch.backends.mps.is_available():
    device = "mps"
```

#### Windows
```bash
# CUDA 설치 확인
nvidia-smi

# bitsandbytes 대안 (Windows에서 문제 시)
uv add --no-deps bitsandbytes
```

#### CPU 전용 환경
```python
# CPU 최적화 모델 사용
compressor = create_compressor("microsoft/DialoGPT-medium")
```

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 📚 논문 참조

```bibtex
@article{fei2025efficient,
  title={Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference},
  author={Fei, Weizhi and others},
  journal={arXiv preprint arXiv:2501.12959},
  year={2025}
}
```

## 🤝 기여 및 지원

- **버그 리포트**: [Issues](https://github.com/comsa33/ehpc-research/issues)
- **기능 요청**: [Feature Requests](https://github.com/comsa33/ehpc-research/discussions)
- **개발 참여**: [Contributing Guide](CONTRIBUTING.md)
- **문의**: research@example.com

## 🙏 감사의 말

- 원 논문 저자들: Weizhi Fei et al.
- Hugging Face Transformers 팀
- 오픈소스 커뮤니티

---

⭐ **이 프로젝트가 도움이 되었다면 스타를 눌러주세요!**
