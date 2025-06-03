# EHPC: Evaluator Head-based Prompt Compression 재현 구현체

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)

"Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference" 논문의 핵심 아이디어를 구현한 순수 Python 라이브러리입니다.

## 🎯 핵심 기능

- **🔍 Evaluator Heads 자동 발견**: Needle-in-a-Haystack 테스트로 중요 정보를 식별하는 attention head 찾기
- **🗜️ 지능형 프롬프트 압축**: attention 기반 토큰 중요도 계산으로 정확한 압축
- **📊 종합적 벤치마크**: SQuAD QA, CNN/DailyMail 요약 등 다양한 태스크 평가
- **🎨 직관적 시각화**: attention 패턴과 압축 결과의 인터랙티브 시각화
- **🌐 웹 데모**: Streamlit 기반 실시간 테스트 인터페이스

## 🚀 빠른 시작

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/ehpc-reproduction.git
cd ehpc-reproduction

# 의존성 설치
uv sync
```

### 기본 사용법

```python
from core.prompt_compressor import EHPCCompressor

# 1. 압축기 초기화
compressor = EHPCCompressor("microsoft/DialoGPT-medium")
compressor.initialize()  # Evaluator Heads 자동 발견

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

### 웹 데모 실행

```bash
# Streamlit 앱 실행
uv run streamlit run demo/streamlit_app.py

# 브라우저에서 자동으로 열립니다!
```

## 📚 논문 참조

```bibtex
@article{wei2025efficient,
  title={Efficient Prompt Compression with Evaluator Heads for Long-Context Transformer Inference},
  author={Wei, Fei and others},
  journal={arXiv preprint arXiv:2501.12959},
  year={2025}
}
```