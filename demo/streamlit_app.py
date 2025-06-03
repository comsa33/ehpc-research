import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time
import torch
import gc

from core.prompt_compressor import EHPCCompressor
from evaluation.benchmarks import EHPCBenchmark
from visualization.attention_viz import AttentionVisualizer

# 페이지 설정
st.set_page_config(
    page_title="EHPC Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# GPU 메모리 정리 함수
def clear_gpu_memory():
    """GPU 메모리 정리"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# 싱글턴 패턴으로 모델 관리
@st.cache_resource
def get_compressor(model_name: str):
    """모델을 한 번만 로딩하는 싱글턴 패턴"""
    clear_gpu_memory()  # 초기화 전 메모리 정리
    return EHPCCompressor(model_name)

# 세션 상태 초기화 (더 안전한 방식)
def init_session_state():
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "evaluator_heads" not in st.session_state:
        st.session_state.evaluator_heads = []

def main():
    st.title("🚀 EHPC: Evaluator Head-based Prompt Compression")
    st.markdown("논문 'Efficient Prompt Compression with Evaluator Heads' 의 구현체")
    
    init_session_state()

    # 사이드바 - 설정
    with st.sidebar:
        st.header("⚙️ 설정")

        model_name = st.selectbox(
            "모델 선택",
            [
                "microsoft/DialoGPT-medium",
                "microsoft/DialoGPT-small", 
                "gpt2",
                "gpt2-medium",
            ],
            help="attention weights에 접근 가능한 HuggingFace 모델",
        )

        max_layers = st.slider("검사할 레이어 수", 1, 6, 3)
        heads_per_layer = st.slider("레이어당 헤드 수", 1, 4, 2)
        
        # GPU 상태 표시
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_allocated = torch.cuda.memory_allocated(0) / 1024**3
            gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
            
            st.info(f"""
            **GPU 상태**
            - 디바이스: {torch.cuda.get_device_name(0)}
            - 총 메모리: {gpu_memory:.1f}GB
            - 사용중: {gpu_allocated:.1f}GB
            - 캐시됨: {gpu_cached:.1f}GB
            """)
        
        # 메모리 정리 버튼
        if st.button("🧹 GPU 메모리 정리"):
            clear_gpu_memory()
            st.success("✅ GPU 메모리가 정리되었습니다!")
            st.rerun()

        if st.button("🔧 시스템 초기화", type="primary"):
            with st.spinner("Evaluator Heads를 찾는 중..."):
                try:
                    # 모델 변경 시 기존 모델 정리
                    if (st.session_state.current_model is not None and 
                        st.session_state.current_model != model_name):
                        clear_gpu_memory()
                        st.cache_resource.clear()  # 캐시된 리소스 정리
                    
                    # 환경 변수 설정으로 CUDA 디버깅 활성화
                    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
                    
                    # 싱글턴 패턴으로 모델 로딩
                    compressor = get_compressor(model_name)
                    
                    # 초기화 수행
                    evaluator_heads = compressor.initialize(
                        max_layers=max_layers, heads_per_layer=heads_per_layer
                    )
                    
                    # 세션 상태 업데이트
                    st.session_state.current_model = model_name
                    st.session_state.initialized = True
                    st.session_state.evaluator_heads = evaluator_heads
                    
                    st.success(f"✅ {len(evaluator_heads)}개의 Evaluator Heads 발견!")

                    # 발견된 헤드들 표시
                    for head in evaluator_heads:
                        st.info(
                            f"Layer {head.layer}, Head {head.head} (신뢰도: {head.confidence_score:.3f})"
                        )

                except Exception as e:
                    st.error(f"❌ 초기화 실패: {e}")
                    # 에러 발생 시 메모리 정리
                    clear_gpu_memory()
                    # 상세 에러 정보 표시
                    if "CUDA" in str(e):
                        st.error("""
                        **CUDA 에러 해결 방법:**
                        1. 'GPU 메모리 정리' 버튼 클릭
                        2. 더 작은 모델 선택 (DialoGPT-small)
                        3. 브라우저 새로고침 후 재시도
                        """)

    # 메인 컨텐츠
    if not st.session_state.initialized:
        st.warning("⚠️ 먼저 사이드바에서 시스템을 초기화해주세요!")
        
        # 초기화 안내
        st.info("""
        **📋 사용 방법:**
        1. 사이드바에서 모델과 설정을 선택
        2. '시스템 초기화' 버튼을 클릭하여 Evaluator Heads 발견
        3. 각 탭에서 기능 테스트
        
        **⚠️ GPU 메모리 부족 시:**
        - 더 작은 모델 (DialoGPT-small) 선택
        - 'GPU 메모리 정리' 버튼 사용
        - 브라우저 새로고침
        """)
        return

    # 탭 구성
    tab1, tab2, tab3 = st.tabs(["💬 프롬프트 압축", "📊 벤치마크", "🔍 시각화"])

    with tab1:
        prompt_compression_tab()

    with tab2:
        benchmark_tab()

    with tab3:
        visualization_tab()


def prompt_compression_tab():
    st.header("💬 프롬프트 압축 및 생성")

    # 프롬프트 입력
    prompt = st.text_area(
        "압축할 프롬프트를 입력하세요:",
        value="""You are a helpful AI assistant with expertise in machine learning and natural language processing. Please analyze the following research paper abstract carefully and provide a comprehensive summary of the key contributions, methodology, and potential applications. Pay special attention to any novel techniques or significant improvements over existing methods. Your analysis should be detailed and technically accurate while remaining accessible to researchers in the field.""",
        height=150,
    )

    col1, col2 = st.columns(2)
    with col1:
        compression_ratio = st.slider("압축 비율", 0.1, 0.9, 0.3, 0.1)
    with col2:
        max_new_tokens = st.slider("생성할 최대 토큰 수", 50, 200, 100)

    if st.button("🚀 압축 및 생성 실행", type="primary"):
        with st.spinner("처리 중..."):
            try:
                # 현재 모델의 압축기 가져오기
                compressor = get_compressor(st.session_state.current_model)
                
                start_time = time.time()
                result = compressor.compress_and_generate(
                    prompt,
                    compression_ratio=compression_ratio,
                    max_new_tokens=max_new_tokens,
                )
                processing_time = time.time() - start_time

                # 결과 표시
                st.success(f"✅ 완료! (처리 시간: {processing_time:.2f}초)")

                # 메트릭 표시
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("압축률", f"{result['compression_ratio']:.1%}")
                with col2:
                    st.metric("원본 길이", f"{result['original_length']} 단어")
                with col3:
                    st.metric("압축 길이", f"{result['compressed_length']} 단어")
                with col4:
                    st.metric(
                        "토큰 절약",
                        f"{result['tokens_total'] - result['tokens_kept']} 개",
                    )

                # 원본 vs 압축 비교
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("📄 원본 프롬프트")
                    st.text_area(
                        "원본 텍스트",
                        value=result["original_text"],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    st.subheader("🤖 원본 응답")
                    st.text_area(
                        "원본 응답",
                        value=result["original_response"],
                        height=150,
                        disabled=True,
                        label_visibility="collapsed",
                    )

                with col2:
                    st.subheader("📄 압축된 프롬프트")
                    st.text_area(
                        "압축된 텍스트",
                        value=result["compressed_text"],
                        height=200,
                        disabled=True,
                        label_visibility="collapsed",
                    )
                    st.subheader("🤖 압축된 응답")
                    st.text_area(
                        "압축된 응답",
                        value=result["compressed_response"],
                        height=150,
                        disabled=True,
                        label_visibility="collapsed",
                    )

            except Exception as e:
                st.error(f"❌ 처리 실패: {e}")
                if "CUDA" in str(e) or "memory" in str(e).lower():
                    if st.button("🧹 에러 후 메모리 정리"):
                        clear_gpu_memory()
                        st.rerun()


def benchmark_tab():
    st.header("📊 벤치마크 평가")

    st.info("다양한 압축 비율에서 모델 성능을 평가합니다.")

    col1, col2 = st.columns(2)
    with col1:
        benchmark_type = st.selectbox(
            "벤치마크 유형", ["QA (SQuAD)", "요약 (CNN/DailyMail)"]
        )
    with col2:
        num_samples = st.slider("샘플 수", 10, 100, 20)

    compression_ratios = st.multiselect(
        "테스트할 압축 비율", [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], default=[0.3, 0.5, 0.7]
    )

    if st.button("🧪 벤치마크 실행"):
        if not compression_ratios:
            st.warning("압축 비율을 최소 하나 선택해주세요!")
            return

        with st.spinner("벤치마크 실행 중... (몇 분 소요될 수 있습니다)"):
            try:
                # 현재 모델의 압축기 가져오기
                compressor = get_compressor(st.session_state.current_model)
                benchmark = EHPCBenchmark(compressor)

                if benchmark_type == "QA (SQuAD)":
                    results = benchmark.run_qa_benchmark(
                        num_samples=num_samples, compression_ratios=compression_ratios
                    )
                else:
                    results = benchmark.run_summarization_benchmark(
                        num_samples=num_samples, compression_ratios=compression_ratios
                    )

                # 결과 표시
                st.success("✅ 벤치마크 완료!")

                # 결과 테이블
                st.subheader("📈 성능 결과")

                import pandas as pd

                df_data = []
                for ratio, metrics in results.items():
                    row = {"압축률": f"{ratio:.1%}"}
                    row.update(
                        {
                            k.upper(): f"{v:.3f}"
                            for k, v in metrics.items()
                            if k != "samples"
                        }
                    )
                    df_data.append(row)

                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True)

                # 시각화
                visualizer = AttentionVisualizer()
                fig = visualizer.plot_benchmark_results(results)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"❌ 벤치마크 실패: {e}")
                if "CUDA" in str(e) or "memory" in str(e).lower():
                    if st.button("🧹 벤치마크 에러 후 메모리 정리"):
                        clear_gpu_memory()
                        st.rerun()


def visualization_tab():
    st.header("🔍 Attention 시각화")

    st.info("프롬프트를 입력하여 토큰 중요도와 attention 패턴을 시각화합니다.")

    viz_prompt = st.text_area(
        "시각화할 텍스트:",
        value="The quick brown fox jumps over the lazy dog in the beautiful garden.",
        height=100,
    )

    viz_ratio = st.slider("시각화용 압축 비율", 0.1, 0.9, 0.5, 0.1)

    if st.button("🎨 시각화 생성"):
        with st.spinner("분석 중..."):
            try:
                # 현재 모델의 압축기 가져오기
                compressor = get_compressor(st.session_state.current_model)
                
                # 압축 결과 얻기
                compression_result = compressor.compress_prompt(
                    viz_prompt, compression_ratio=viz_ratio
                )

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
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("총 토큰", len(compression_result.original_tokens))
                with col2:
                    st.metric("선택된 토큰", len(compression_result.selected_indices))
                with col3:
                    st.metric(
                        "실제 압축률", f"{compression_result.compression_ratio:.1%}"
                    )

                # 압축 결과 텍스트
                compressed_text = compressor.tokens_to_text(
                    compression_result.compressed_tokens
                )

                st.subheader("📄 압축 결과")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("원본 텍스트", viz_prompt, height=100, disabled=True, label_visibility="collapsed")
                with col2:
                    st.text_area("압축된 텍스트", compressed_text, height=100, disabled=True, label_visibility="collapsed")

                # 리포트 생성
                report = visualizer.create_compression_report(compression_result)
                with st.expander("📋 상세 리포트"):
                    st.markdown(report)

            except Exception as e:
                st.error(f"❌ 시각화 실패: {e}")
                if "CUDA" in str(e) or "memory" in str(e).lower():
                    if st.button("🧹 시각화 에러 후 메모리 정리"):
                        clear_gpu_memory()
                        st.rerun()


if __name__ == "__main__":
    main()
