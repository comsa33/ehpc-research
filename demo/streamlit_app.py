import gc
import os
import sys
import time
import traceback

import streamlit as st
import torch

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

try:
    from core.evaluator_heads import (
        ModelConfig,
        get_model_info,
        get_recommended_model,
        list_supported_models,
        test_model_loading,
    )
    from core.prompt_compressor import EHPCCompressor, create_compressor
    from evaluation.benchmarks import EHPCBenchmark
    from visualization.attention_viz import AttentionVisualizer
except ImportError as e:
    st.error(f"❌ 모듈 임포트 실패: {e}")
    st.error(
        "프로젝트 루트 디렉토리에서 실행해주세요: `streamlit run demo/streamlit_app.py`"
    )
    st.stop()

# 페이지 설정
st.set_page_config(
    page_title="EHPC Demo - 업그레이드",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS 스타일
st.markdown(
    """
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .error-box {
        background: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)


# GPU 메모리 정리 함수
def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


# 안전한 세션 상태 초기화
def init_session_state():
    """세션 상태 안전 초기화"""
    defaults = {
        "current_model": None,
        "initialized": False,
        "evaluator_heads": [],
        "compressor": None,
        "model_stats": {},
        "last_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# 모델 로딩 함수 (캐시된)
@st.cache_resource
def load_compressor(model_name: str, use_auto_init: bool = False):
    """모델 로딩 (캐시된 리소스)"""
    try:
        clear_gpu_memory()  # 로딩 전 메모리 정리
        compressor = EHPCCompressor(model_name, auto_initialize=use_auto_init)
        return compressor, None
    except Exception as e:
        error_msg = f"모델 로딩 실패: {str(e)}"
        return None, error_msg


def get_hardware_info():
    """하드웨어 정보 수집"""
    info = {"device": "CPU"}

    if torch.cuda.is_available():
        gpu_props = torch.cuda.get_device_properties(0)
        memory_gb = gpu_props.total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated(0) / (1024**3)
        cached_gb = torch.cuda.memory_reserved(0) / (1024**3)

        info.update(
            {
                "device": "CUDA",
                "gpu_name": gpu_props.name,
                "total_memory": memory_gb,
                "allocated_memory": allocated_gb,
                "cached_memory": cached_gb,
            }
        )
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info.update({"device": "MPS (Apple Silicon)", "note": "통합 메모리 사용"})

    return info


def show_hf_auth_status():
    """Hugging Face 인증 상태 표시"""
    import os

    # 토큰 확인
    token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

    if token:
        st.success("🔑 **Hugging Face 인증됨**")
    else:
        st.warning("⚠️ **Hugging Face 미인증**")

        with st.expander("🔐 인증 방법"):
            st.markdown(
                """
            **Gated 모델 접근을 위해 인증이 필요합니다:**
            
            **방법 1: 환경 변수 설정**
            ```bash
            export HUGGINGFACE_TOKEN="your_token_here"
            ```
            
            **방법 2: .env 파일 생성**
            ```bash
            echo "HUGGINGFACE_TOKEN=your_token_here" > .env
            ```
            
            **방법 3: CLI 로그인**
            ```bash
            huggingface-cli login
            ```
            
            **토큰 발급**: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
            """
            )

            # 간단한 토큰 입력 UI
            st.subheader("임시 토큰 입력")
            token_input = st.text_input(
                "Hugging Face 토큰", type="password", help="이 세션에서만 사용됩니다"
            )

            if st.button("토큰 설정") and token_input:
                os.environ["HUGGINGFACE_TOKEN"] = token_input
                st.success("✅ 토큰이 설정되었습니다!")
                st.rerun()


def check_model_access(model_name: str) -> bool:
    """모델 접근 가능성 확인"""
    try:
        import os

        from transformers import AutoTokenizer

        token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        # 간단한 토크나이저 로딩으로 접근 테스트
        AutoTokenizer.from_pretrained(model_name, token=token, trust_remote_code=True)
        return True
    except Exception as e:
        if "gated repo" in str(e).lower():
            return False
        return True  # 다른 오류는 접근 가능한 것으로 간주


def show_model_selection_sidebar():
    """사이드바 모델 선택 UI"""
    with st.sidebar:
        st.header("⚙️ 모델 설정")

        # Hugging Face 인증 상태 확인
        show_hf_auth_status()

        # 하드웨어 정보 표시
        hw_info = get_hardware_info()
        if hw_info["device"] == "CUDA":
            st.success(
                f"""
            **🎮 GPU 정보**
            - {hw_info['gpu_name']}
            - 총 메모리: {hw_info['total_memory']:.1f}GB
            - 사용 중: {hw_info['allocated_memory']:.1f}GB
            - 캐시: {hw_info['cached_memory']:.1f}GB
            """
            )
        elif hw_info["device"] == "MPS":
            st.info(f"**🍎 Apple Silicon**: {hw_info['note']}")
        else:
            st.warning("**💻 CPU 모드**: GPU를 사용할 수 없습니다")

        # 자동 추천
        recommended_model = get_recommended_model()
        st.info(f"🎯 **환경 최적화 추천**: `{recommended_model}`")

        # 모델 카테고리 선택
        model_categories = {
            "🎯 자동 추천": [recommended_model],
            "🏆 최고 성능": [
                "meta-llama/Llama-3.2-3B-Instruct",
                "beomi/KoAlpaca-Polyglot-5.8B",
            ],
            "⚖️ 균형형": ["Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-3.5-mini-instruct"],
            "⚡ 경량형": ["google/gemma-2-2b", "microsoft/DialoGPT-medium"],
            "🇰🇷 한국어 특화": [
                "beomi/KoAlpaca-Polyglot-5.8B",
                "Qwen/Qwen2.5-3B-Instruct",
            ],
        }

        selected_category = st.selectbox(
            "모델 카테고리",
            list(model_categories.keys()),
            help="사용 목적에 따른 모델 분류",
        )

        available_models = model_categories[selected_category]
        selected_model = st.selectbox(
            "모델 선택", available_models, help="선택한 카테고리의 추천 모델들"
        )

        # 모델 정보 표시
        model_info = get_model_info(selected_model)
        if model_info:
            korean_stars = "⭐" * model_info.get("korean_support", 1)

            # 모델 접근 가능성 확인
            access_status = check_model_access(selected_model)
            access_icon = "✅" if access_status else "🔒"

            st.markdown(
                f"""
            **📊 모델 정보** {access_icon}
            - **파라미터**: {model_info.get('params', 'Unknown')}
            - **최소 메모리**: {model_info.get('min_memory_gb', 'Unknown')}GB
            - **컨텍스트 길이**: {model_info.get('context_length', 'Unknown'):,}
            - **한국어 지원**: {korean_stars} ({model_info.get('korean_support', 1)}/5)
            - **장점**: {', '.join(model_info.get('pros', []))}
            """
            )

            if not access_status:
                st.error("🔒 **이 모델은 인증이 필요합니다!**")
                st.info("위의 '🔐 인증 방법'을 참조하여 Hugging Face에 로그인하세요.")

        # 고급 설정
        with st.expander("🔧 고급 설정"):
            max_layers = st.slider(
                "검사할 레이어 수",
                1,
                6,
                3,
                help="더 많은 레이어 검사 시 정확도 향상, 속도 저하",
            )
            heads_per_layer = st.slider(
                "레이어당 헤드 수", 1, 4, 2, help="더 많은 헤드 사용 시 품질 향상"
            )
            auto_initialize = st.checkbox(
                "자동 초기화", True, help="모델 로딩 시 바로 Evaluator Heads 찾기"
            )

        # 초기화 버튼
        col1, col2 = st.columns(2)

        with col1:
            if st.button("🚀 모델 초기화", type="primary", use_container_width=True):
                initialize_model(
                    selected_model, max_layers, heads_per_layer, auto_initialize
                )

        with col2:
            if st.button("🧹 메모리 정리", use_container_width=True):
                clear_gpu_memory()
                if "compressor" in st.session_state:
                    del st.session_state.compressor
                st.cache_resource.clear()
                st.success("✅ 메모리 정리 완료")
                st.rerun()

        return selected_model, max_layers, heads_per_layer


def initialize_model(model_name, max_layers, heads_per_layer, auto_initialize):
    """모델 초기화 처리"""
    with st.spinner(f"🔄 {model_name} 로딩 중..."):
        try:
            # 기존 모델과 다르면 캐시 클리어
            if st.session_state.get("current_model") != model_name:
                clear_gpu_memory()
                st.cache_resource.clear()

            # 모델 로딩
            compressor, error = load_compressor(model_name, use_auto_init=False)

            if error:
                st.error(f"❌ {error}")
                return

            # 초기화 수행
            if auto_initialize:
                with st.spinner("🧠 Evaluator Heads 찾는 중..."):
                    evaluator_heads = compressor.initialize(
                        max_layers=max_layers, heads_per_layer=heads_per_layer
                    )
            else:
                evaluator_heads = []

            # 세션 상태 업데이트
            st.session_state.current_model = model_name
            st.session_state.compressor = compressor
            st.session_state.evaluator_heads = evaluator_heads
            st.session_state.initialized = auto_initialize
            st.session_state.model_stats = compressor.get_compression_stats()
            st.session_state.last_error = None

            # 성공 메시지
            if auto_initialize:
                st.success(
                    f"✅ 모델 로딩 및 초기화 완료! ({len(evaluator_heads)}개 Evaluator Heads 발견)"
                )

                # 발견된 헤드 정보 표시
                if evaluator_heads:
                    st.write("**발견된 Evaluator Heads:**")
                    for i, head in enumerate(evaluator_heads):
                        st.write(
                            f"  {i+1}. Layer {head.layer}, Head {head.head} "
                            f"(신뢰도: {head.confidence_score:.3f})"
                        )
            else:
                st.success("✅ 모델 로딩 완료! 수동 초기화가 필요합니다.")

        except Exception as e:
            error_msg = f"초기화 실패: {str(e)}"
            st.error(f"❌ {error_msg}")
            st.session_state.last_error = error_msg

            # 상세 오류 정보
            if "CUDA" in str(e) or "memory" in str(e).lower():
                st.error(
                    """
                **💡 CUDA/메모리 오류 해결 방법:**
                1. '메모리 정리' 버튼 클릭
                2. 더 작은 모델 선택 (경량형 카테고리)
                3. 브라우저 새로고침 후 재시도
                """
                )


def main():
    """메인 함수"""

    # 헤더
    st.markdown(
        """
    <div class="main-header">
        <h1>🚀 EHPC: Evaluator Head-based Prompt Compression</h1>
        <p>업그레이드된 논문 구현체 - 최신 모델 지원</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # 세션 상태 초기화
    init_session_state()

    # 사이드바
    selected_model, max_layers, heads_per_layer = show_model_selection_sidebar()

    # 메인 컨텐츠
    if not st.session_state.initialized:
        show_welcome_screen()
    else:
        show_main_interface()


def show_welcome_screen():
    """초기 화면"""
    st.markdown(
        """
    ## 👋 EHPC 데모에 오신 것을 환영합니다!
    
    이 업그레이드된 버전은 다음과 같은 개선사항을 제공합니다:
    
    ### 🆕 새로운 기능들
    - **최신 모델 지원**: Llama 3.2, Qwen 2.5, Gemma 2, Phi 3.5 등
    - **한국어 특화**: 한국어 모델 및 테스트 데이터 지원
    - **자동 최적화**: 하드웨어 환경에 맞는 모델 자동 추천
    - **메모리 효율**: 8bit 양자화 및 메모리 관리 개선
    - **안정성 강화**: 오류 처리 및 폴백 시스템
    
    ### 📋 사용 방법
    1. **사이드바**에서 적합한 모델 선택
    2. **모델 초기화** 버튼으로 Evaluator Heads 발견
    3. 각 **탭**에서 다양한 기능 테스트
    
    ### ⚠️ 문제 해결
    - **GPU 메모리 부족**: 경량형 모델 선택 또는 메모리 정리
    - **모델 로딩 실패**: 자동 추천 모델 사용
    - **성능 저하**: 하드웨어에 맞는 카테고리 선택
    
    👈 **시작하려면 사이드바에서 모델을 선택하고 초기화하세요!**
    """
    )

    # 빠른 테스트 버튼
    st.markdown("### 🧪 빠른 테스트")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("⚡ 경량 모델로 빠른 시작", use_container_width=True):
            initialize_model("google/gemma-2-2b", 2, 1, True)

    with col2:
        if st.button("🎯 추천 모델로 시작", use_container_width=True):
            recommended = get_recommended_model()
            initialize_model(recommended, 3, 2, True)

    with col3:
        if st.button("🇰🇷 한국어 모델로 시작", use_container_width=True):
            initialize_model("Qwen/Qwen2.5-3B-Instruct", 3, 2, True)


def show_main_interface():
    """메인 인터페이스"""
    # 현재 모델 정보 표시
    if st.session_state.model_stats:
        stats = st.session_state.model_stats
        model_info = stats.get("model_info", {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("현재 모델", stats.get("model_name", "Unknown").split("/")[-1])
        with col2:
            st.metric("파라미터", model_info.get("parameters", "Unknown"))
        with col3:
            st.metric("Evaluator Heads", stats.get("num_evaluator_heads", 0))
        with col4:
            st.metric("디바이스", model_info.get("device", "Unknown"))

    # 탭 구성
    tab1, tab2, tab3, tab4 = st.tabs(
        ["💬 프롬프트 압축", "📊 벤치마크", "🔍 시각화", "📈 모델 정보"]
    )

    with tab1:
        prompt_compression_tab()

    with tab2:
        benchmark_tab()

    with tab3:
        visualization_tab()

    with tab4:
        model_info_tab()


def prompt_compression_tab():
    """프롬프트 압축 탭"""
    st.header("💬 프롬프트 압축 및 생성")

    # 예시 프롬프트 선택
    example_prompts = {
        "영어 예시": """You are a helpful AI assistant with expertise in machine learning and natural language processing. Please analyze the following research paper abstract carefully and provide a comprehensive summary of the key contributions, methodology, and potential applications. Pay special attention to any novel techniques or significant improvements over existing methods.""",
        "한국어 예시": """인공지능과 자연어 처리 분야의 전문가로서, 다음 연구 논문의 초록을 신중하게 분석하고 주요 기여점, 방법론, 그리고 잠재적 응용 분야에 대한 포괄적인 요약을 제공해 주세요. 새로운 기술이나 기존 방법 대비 중요한 개선 사항에 특별히 주의를 기울여 주세요.""",
        "기술 문서": """The transformer architecture has revolutionized natural language processing through its attention mechanism. This technology enables models to focus on relevant parts of the input sequence when generating outputs. Key innovations include multi-head attention, positional encoding, and layer normalization. These components work together to achieve state-of-the-art performance across various NLP tasks.""",
        "사용자 정의": "",
    }

    selected_example = st.selectbox("예시 선택", list(example_prompts.keys()))

    if selected_example == "사용자 정의":
        prompt = st.text_area("압축할 프롬프트를 입력하세요:", height=150)
    else:
        prompt = st.text_area(
            "압축할 프롬프트를 입력하세요:",
            value=example_prompts[selected_example],
            height=150,
        )

    # 설정
    col1, col2, col3 = st.columns(3)
    with col1:
        compression_ratio = st.slider(
            "압축 비율", 0.1, 0.9, 0.3, 0.1, help="유지할 토큰의 비율"
        )
    with col2:
        max_new_tokens = st.slider(
            "생성할 최대 토큰 수", 50, 300, 100, help="응답 길이 제한"
        )
    with col3:
        temperature = st.slider(
            "생성 온도", 0.1, 1.0, 0.7, 0.1, help="응답의 창의성 조절"
        )

    # 고급 옵션
    with st.expander("🔧 고급 옵션"):
        preserve_special = st.checkbox(
            "특수 토큰 보존", True, help="[CLS], <s> 등 특수 토큰 유지"
        )
        min_tokens = st.number_input(
            "최소 토큰 수", 3, 50, 5, help="압축 후 최소 유지할 토큰 수"
        )

    if st.button("🚀 압축 및 생성 실행", type="primary"):
        if not prompt.strip():
            st.warning("⚠️ 프롬프트를 입력해주세요!")
            return

        with st.spinner("🔄 처리 중..."):
            try:
                compressor = st.session_state.compressor

                start_time = time.time()
                result = compressor.compress_and_generate(
                    prompt,
                    compression_ratio=compression_ratio,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                processing_time = time.time() - start_time

                # 결과 표시
                st.markdown(
                    '<div class="success-box">✅ 압축 및 생성 완료!</div>',
                    unsafe_allow_html=True,
                )

                # 메트릭 표시
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("압축률", f"{result['compression_ratio']:.1%}")
                with col2:
                    st.metric("원본 길이", f"{result['original_length']} 단어")
                with col3:
                    st.metric("압축 길이", f"{result['compressed_length']} 단어")
                with col4:
                    st.metric(
                        "토큰 절약", f"{result['tokens_total'] - result['tokens_kept']}"
                    )
                with col5:
                    st.metric("처리 시간", f"{processing_time:.2f}초")

                # 원본 vs 압축 비교
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📄 원본")
                    st.text_area(
                        "원본 프롬프트",
                        result["original_text"],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    st.subheader("🤖 원본 응답")
                    st.text_area(
                        "원본 응답",
                        result["original_response"],
                        height=150,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    st.caption(
                        f"생성 시간: {result.get('generation_time_original', 0):.2f}초"
                    )

                with col2:
                    st.subheader("📄 압축됨")
                    st.text_area(
                        "압축 프롬프트",
                        result["compressed_text"],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    st.subheader("🤖 압축 응답")
                    st.text_area(
                        "압축 응답",
                        result["compressed_response"],
                        height=150,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    st.caption(
                        f"생성 시간: {result.get('generation_time_compressed', 0):.2f}초"
                    )

                # 추가 분석
                with st.expander("📊 상세 분석"):
                    st.json(
                        {
                            "압축 통계": {
                                "원본 토큰 수": result["tokens_total"],
                                "압축 토큰 수": result["tokens_kept"],
                                "제거된 토큰": result["tokens_total"]
                                - result["tokens_kept"],
                                "압축률": f"{result['compression_ratio']:.1%}",
                            },
                            "성능 분석": {
                                "원본 입력 토큰": result.get(
                                    "original_tokens_count", "Unknown"
                                ),
                                "압축 입력 토큰": result.get(
                                    "compressed_tokens_count", "Unknown"
                                ),
                                "원본 생성 시간": f"{result.get('generation_time_original', 0):.3f}초",
                                "압축 생성 시간": f"{result.get('generation_time_compressed', 0):.3f}초",
                            },
                        }
                    )

            except Exception as e:
                st.error(f"❌ 처리 실패: {str(e)}")

                # 오류 유형별 해결책 제안
                error_str = str(e).lower()
                if "cuda" in error_str or "memory" in error_str:
                    st.error(
                        """
                    **💡 메모리 오류 해결 방법:**
                    1. 사이드바에서 '메모리 정리' 버튼 클릭
                    2. 더 낮은 압축 비율 시도 (0.5 이상)
                    3. 더 작은 모델로 변경
                    """
                    )
                elif "token" in error_str:
                    st.error(
                        """
                    **💡 토큰 오류 해결 방법:**
                    1. 더 짧은 프롬프트 사용
                    2. 최소 토큰 수 줄이기
                    3. 다른 예시 프롬프트 시도
                    """
                    )


def benchmark_tab():
    """벤치마크 탭"""
    st.header("📊 벤치마크 평가")

    st.info("다양한 압축 비율에서 모델 성능을 평가합니다.")

    # 벤치마크 설정
    col1, col2 = st.columns(2)
    with col1:
        benchmark_type = st.selectbox(
            "벤치마크 유형",
            ["QA (SQuAD)", "요약 (CNN/DailyMail)", "압축 성능 테스트"],
            help="평가할 태스크 선택",
        )
    with col2:
        num_samples = st.slider("샘플 수", 5, 50, 10, help="테스트할 샘플 개수")

    compression_ratios = st.multiselect(
        "테스트할 압축 비율",
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        default=[0.3, 0.5, 0.7],
        help="여러 압축 비율에서 성능 비교",
    )

    if st.button("🧪 벤치마크 실행"):
        if not compression_ratios:
            st.warning("⚠️ 압축 비율을 최소 하나 선택해주세요!")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            compressor = st.session_state.compressor

            if benchmark_type == "압축 성능 테스트":
                # 압축 성능 자체 테스트
                test_texts = [
                    "인공지능은 현대 기술의 핵심입니다. 머신러닝과 딥러닝을 통해 다양한 문제를 해결하고 있습니다.",
                    "Natural language processing has revolutionized how we interact with computers and analyze text data.",
                    "The attention mechanism in transformers allows models to focus on relevant parts of the input sequence.",
                    "한국어 자연어 처리는 형태소 분석, 구문 분석, 의미 분석 등 다양한 단계를 포함합니다.",
                    "Large language models like GPT and BERT have achieved remarkable performance across various NLP tasks.",
                ]

                status_text.text("압축 성능 테스트 실행 중...")
                results = compressor.benchmark_compression(
                    test_texts, compression_ratios
                )

                # 결과 표시
                st.success("✅ 압축 벤치마크 완료!")

                # 결과 테이블
                import pandas as pd

                df_data = []
                for ratio, metrics in results.items():
                    df_data.append(
                        {
                            "압축률": f"{float(ratio):.1%}",
                            "평균 실제 압축률": f"{metrics['avg_actual_ratio']:.1%}",
                            "평균 처리 시간": f"{metrics['avg_compression_time']:.3f}초",
                            "테스트 샘플": len(metrics["results"]),
                        }
                    )

                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)

                # 상세 결과
                with st.expander("📈 상세 결과"):
                    for ratio, metrics in results.items():
                        st.write(f"**압축률 {float(ratio):.1%} 결과:**")
                        for i, result in enumerate(metrics["results"]):
                            st.write(
                                f"  샘플 {i+1}: {result['original_length']}→{result['compressed_length']} 토큰 "
                                f"({result['actual_ratio']:.1%}, {result['compression_time']:.3f}초)"
                            )

            else:
                # 기존 벤치마크 (QA, 요약)
                with st.spinner("벤치마크 실행 중... (몇 분 소요될 수 있습니다)"):
                    benchmark = EHPCBenchmark(compressor)

                    if benchmark_type == "QA (SQuAD)":
                        status_text.text("SQuAD QA 벤치마크 실행 중...")
                        results = benchmark.run_qa_benchmark(
                            num_samples=num_samples,
                            compression_ratios=compression_ratios,
                        )
                    else:
                        status_text.text("CNN/DailyMail 요약 벤치마크 실행 중...")
                        results = benchmark.run_summarization_benchmark(
                            num_samples=num_samples,
                            compression_ratios=compression_ratios,
                        )

                    progress_bar.progress(100)

                    # 결과 표시
                    st.success("✅ 벤치마크 완료!")

                    # 결과 테이블
                    import pandas as pd

                    df_data = []
                    for ratio, metrics in results.items():
                        row = {"압축률": f"{float(ratio):.1%}"}
                        row.update(
                            {
                                k.upper(): f"{v:.3f}"
                                for k, v in metrics.items()
                                if k != "samples" and isinstance(v, (int, float))
                            }
                        )
                        row["샘플 수"] = metrics.get("samples", "Unknown")
                        df_data.append(row)

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)

                    # 시각화
                    if len(results) > 1:
                        try:
                            visualizer = AttentionVisualizer()
                            fig = visualizer.plot_benchmark_results(results)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ 시각화 생성 실패: {e}")

        except Exception as e:
            st.error(f"❌ 벤치마크 실패: {str(e)}")
            progress_bar.empty()
            status_text.empty()


def visualization_tab():
    """시각화 탭"""
    st.header("🔍 Attention 시각화")

    st.info("프롬프트를 입력하여 토큰 중요도와 attention 패턴을 시각화합니다.")

    # 예시 텍스트 선택
    viz_examples = {
        "영어 예시": "The quick brown fox jumps over the lazy dog in the beautiful garden.",
        "한국어 예시": "빠른 갈색 여우가 아름다운 정원에서 게으른 개를 뛰어넘습니다.",
        "기술 텍스트": "Machine learning algorithms analyze patterns in data to make predictions.",
        "사용자 정의": "",
    }

    selected_viz_example = st.selectbox("시각화 예시 선택", list(viz_examples.keys()))

    if selected_viz_example == "사용자 정의":
        viz_prompt = st.text_area("시각화할 텍스트:", height=100)
    else:
        viz_prompt = st.text_area(
            "시각화할 텍스트:", value=viz_examples[selected_viz_example], height=100
        )

    viz_ratio = st.slider("시각화용 압축 비율", 0.1, 0.9, 0.5, 0.1)

    if st.button("🎨 시각화 생성"):
        if not viz_prompt.strip():
            st.warning("⚠️ 시각화할 텍스트를 입력해주세요!")
            return

        with st.spinner("🔍 분석 중..."):
            try:
                compressor = st.session_state.compressor

                # 압축 결과 얻기
                compression_result = compressor.compress_prompt(
                    viz_prompt, compression_ratio=viz_ratio
                )

                # 시각화 생성
                visualizer = AttentionVisualizer()

                # 토큰 중요도 시각화
                fig_importance = visualizer.plot_token_importance(
                    compression_result.original_tokens,
                    compression_result.token_scores,
                    compression_result.selected_indices,
                    title="토큰 중요도 점수 (빨간색: 선택된 토큰)",
                )
                st.plotly_chart(fig_importance, use_container_width=True)

                # 압축 통계
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("총 토큰", len(compression_result.original_tokens))
                with col2:
                    st.metric("선택된 토큰", len(compression_result.selected_indices))
                with col3:
                    st.metric(
                        "실제 압축률", f"{compression_result.compression_ratio:.1%}"
                    )
                with col4:
                    st.metric(
                        "Evaluator Heads", len(compression_result.evaluator_heads)
                    )

                # 압축 결과 텍스트 비교
                compressed_text = compressor.tokens_to_text(
                    compression_result.compressed_tokens
                )

                st.subheader("📄 압축 결과 비교")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area(
                        "원본 텍스트",
                        viz_prompt,
                        height=100,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                with col2:
                    st.text_area(
                        "압축된 텍스트",
                        compressed_text,
                        height=100,
                        disabled=True,
                        label_visibility="collapsed",
                    )

                # 토큰별 상세 정보
                with st.expander("🔍 토큰별 상세 분석"):
                    df_data = []
                    for i, (token, score) in enumerate(
                        zip(
                            compression_result.original_tokens,
                            compression_result.token_scores,
                        )
                    ):
                        df_data.append(
                            {
                                "위치": i,
                                "토큰": token,
                                "중요도 점수": f"{score:.4f}",
                                "선택됨": (
                                    "✅"
                                    if i in compression_result.selected_indices
                                    else "❌"
                                ),
                            }
                        )

                    import pandas as pd

                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)

                # 리포트 생성
                try:
                    report = visualizer.create_compression_report(compression_result)
                    with st.expander("📋 상세 리포트"):
                        st.markdown(report)
                except Exception as e:
                    st.warning(f"⚠️ 리포트 생성 실패: {e}")

            except Exception as e:
                st.error(f"❌ 시각화 실패: {str(e)}")


def model_info_tab():
    """모델 정보 탭"""
    st.header("📈 모델 정보 및 통계")

    if st.session_state.model_stats:
        stats = st.session_state.model_stats
        model_info = stats.get("model_info", {})

        # 기본 정보
        st.subheader("📊 기본 정보")
        col1, col2 = st.columns(2)

        with col1:
            st.info(
                f"""
            **모델 이름**: {stats.get('model_name', 'Unknown')}
            **파라미터**: {model_info.get('parameters', 'Unknown')}
            **디바이스**: {model_info.get('device', 'Unknown')}
            **양자화**: {'✅' if model_info.get('quantized', False) else '❌'}
            """
            )

        with col2:
            st.info(
                f"""
            **레이어 수**: {model_info.get('num_layers', 'Unknown')}
            **Attention Heads**: {model_info.get('num_heads', 'Unknown')}
            **컨텍스트 길이**: {model_info.get('context_length', 'Unknown'):,}
            **한국어 지원**: {'⭐' * model_info.get('korean_support', 1)} ({model_info.get('korean_support', 1)}/5)
            """
            )

        # Evaluator Heads 정보
        if stats.get("evaluator_heads"):
            st.subheader("🧠 Evaluator Heads 정보")

            heads_data = []
            for head in stats["evaluator_heads"]:
                heads_data.append(
                    {
                        "레이어": head["layer"],
                        "헤드": head["head"],
                        "신뢰도": f"{head['confidence']:.4f}",
                        "선택성": f"{head['selectivity']:.4f}",
                        "종합 점수": f"{head['confidence'] + (1-head['selectivity']):.4f}",
                    }
                )

            import pandas as pd

            df = pd.DataFrame(heads_data)
            st.dataframe(df, use_container_width=True)

            # 헤드 분포 시각화
            try:
                import plotly.express as px

                layer_counts = {}
                for head in stats["evaluator_heads"]:
                    layer = head["layer"]
                    layer_counts[layer] = layer_counts.get(layer, 0) + 1

                if layer_counts:
                    fig = px.bar(
                        x=list(layer_counts.keys()),
                        y=list(layer_counts.values()),
                        title="레이어별 Evaluator Heads 분포",
                        labels={"x": "레이어", "y": "헤드 수"},
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.warning(f"⚠️ 시각화 생성 실패: {e}")

        # 메모리 사용량
        st.subheader("💾 메모리 사용량")
        memory_info = model_info.get("memory_usage", "Unknown")
        st.code(memory_info)

        # 모델 장단점
        if model_info.get("pros"):
            st.subheader("✅ 모델 장점")
            for pro in model_info["pros"]:
                st.write(f"• {pro}")

        # 하드웨어 정보
        st.subheader("🖥️ 하드웨어 정보")
        hw_info = get_hardware_info()
        st.json(hw_info)

        # 수동 초기화 옵션
        if not st.session_state.initialized:
            st.subheader("🔧 수동 초기화")
            st.warning("⚠️ Evaluator Heads가 아직 초기화되지 않았습니다.")

            col1, col2 = st.columns(2)
            with col1:
                manual_layers = st.slider("레이어 수", 1, 6, 3, key="manual_layers")
            with col2:
                manual_heads = st.slider("헤드 수", 1, 4, 2, key="manual_heads")

            if st.button("🚀 수동 초기화 실행", type="primary"):
                with st.spinner("🧠 Evaluator Heads 찾는 중..."):
                    try:
                        compressor = st.session_state.compressor
                        evaluator_heads = compressor.initialize(
                            max_layers=manual_layers, heads_per_layer=manual_heads
                        )

                        st.session_state.evaluator_heads = evaluator_heads
                        st.session_state.initialized = True
                        st.session_state.model_stats = (
                            compressor.get_compression_stats()
                        )

                        st.success(
                            f"✅ 초기화 완료! {len(evaluator_heads)}개 Evaluator Heads 발견"
                        )
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ 초기화 실패: {e}")

    else:
        st.warning(
            "⚠️ 모델이 로드되지 않았습니다. 사이드바에서 모델을 선택하고 초기화해주세요."
        )

        # 지원 모델 리스트 표시
        st.subheader("📋 지원되는 모델들")

        models_data = []
        for model_name in list_supported_models():
            info = get_model_info(model_name)
            models_data.append(
                {
                    "모델명": model_name,
                    "파라미터": info.get("params", "Unknown"),
                    "최소 메모리": f"{info.get('min_memory_gb', 'Unknown')}GB",
                    "한국어 지원": "⭐" * info.get("korean_support", 1),
                    "장점": ", ".join(info.get("pros", [])[:2]),  # 처음 2개만
                }
            )

        import pandas as pd

        df = pd.DataFrame(models_data)
        st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ 애플리케이션 오류: {str(e)}")
        st.error("스택 트레이스:")
        st.code(traceback.format_exc())

        if st.button("🔄 페이지 새로고침"):
            st.rerun()
