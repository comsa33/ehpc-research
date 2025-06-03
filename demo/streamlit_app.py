import os
import sys

import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import time

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

# 세션 상태 초기화
if "compressor" not in st.session_state:
    st.session_state.compressor = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False


def main():
    st.title("🚀 EHPC: Evaluator Head-based Prompt Compression")
    st.markdown("논문 'Efficient Prompt Compression with Evaluator Heads' 의 구현체")

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

        if st.button("🔧 시스템 초기화", type="primary"):
            with st.spinner("Evaluator Heads를 찾는 중..."):
                try:
                    st.session_state.compressor = EHPCCompressor(model_name)
                    evaluator_heads = st.session_state.compressor.initialize(
                        max_layers=max_layers, heads_per_layer=heads_per_layer
                    )
                    st.session_state.initialized = True
                    st.success(f"✅ {len(evaluator_heads)}개의 Evaluator Heads 발견!")

                    # 발견된 헤드들 표시
                    for head in evaluator_heads:
                        st.info(
                            f"Layer {head.layer}, Head {head.head} (신뢰도: {head.confidence_score:.3f})"
                        )

                except Exception as e:
                    st.error(f"❌ 초기화 실패: {e}")

    # 메인 컨텐츠
    if not st.session_state.initialized:
        st.warning("⚠️ 먼저 사이드바에서 시스템을 초기화해주세요!")
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
                start_time = time.time()
                result = st.session_state.compressor.compress_and_generate(
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
                benchmark = EHPCBenchmark(st.session_state.compressor)

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
                # 압축 결과 얻기
                compression_result = st.session_state.compressor.compress_prompt(
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
                compressed_text = st.session_state.compressor.tokens_to_text(
                    compression_result.compressed_tokens
                )

                st.subheader("📄 압축 결과")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("원본", viz_prompt, height=100, disabled=True)
                with col2:
                    st.text_area("압축됨", compressed_text, height=100, disabled=True)

                # 리포트 생성
                report = visualizer.create_compression_report(compression_result)
                with st.expander("📋 상세 리포트"):
                    st.markdown(report)

            except Exception as e:
                st.error(f"❌ 시각화 실패: {e}")


if __name__ == "__main__":
    main()
