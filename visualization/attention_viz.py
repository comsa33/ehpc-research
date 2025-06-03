from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


class AttentionVisualizer:
    """Attention과 압축 결과를 시각화하는 클래스"""

    def __init__(self):
        plt.style.use("default")
        sns.set_palette("husl")

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        title: str = "Attention Heatmap",
    ) -> go.Figure:
        """Attention weights 히트맵"""
        fig = go.Figure(
            data=go.Heatmap(
                z=attention_weights,
                x=tokens,
                y=tokens,
                colorscale="Blues",
                showscale=True,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            width=800,
            height=600,
        )

        return fig

    def plot_token_importance(
        self,
        tokens: List[str],
        scores: np.ndarray,
        selected_indices: List[int],
        title: str = "Token Importance Scores",
    ) -> go.Figure:
        """토큰 중요도 점수 시각화"""
        colors = [
            "red" if i in selected_indices else "lightblue" for i in range(len(tokens))
        ]

        fig = go.Figure(
            data=go.Bar(
                x=list(range(len(tokens))),
                y=scores,
                text=tokens,
                textposition="outside",
                marker_color=colors,
                hovertemplate="Token: %{text}<br>Score: %{y:.3f}<extra></extra>",
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Token Position",
            yaxis_title="Importance Score",
            showlegend=False,
            height=500,
        )

        return fig

    def plot_compression_comparison(
        self, compression_results: Dict[str, Any]
    ) -> go.Figure:
        """압축 전후 비교 시각화"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Length Comparison",
                "Compression Ratios",
                "Processing Time",
                "Quality Metrics",
            ),
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "bar"}, {"type": "scatter"}],
            ],
        )

        # 길이 비교
        fig.add_trace(
            go.Bar(
                x=["Original", "Compressed"],
                y=[
                    compression_results["original_length"],
                    compression_results["compressed_length"],
                ],
                name="Length",
            ),
            row=1,
            col=1,
        )

        # 압축 비율 파이 차트
        kept_ratio = compression_results["compression_ratio"]
        removed_ratio = 1 - kept_ratio

        fig.add_trace(
            go.Pie(
                labels=["Kept", "Removed"],
                values=[kept_ratio, removed_ratio],
                name="Compression",
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            height=800, showlegend=False, title_text="Compression Analysis Dashboard"
        )

        return fig

    def plot_benchmark_results(
        self, benchmark_results: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """벤치마크 결과 시각화"""
        compression_ratios = list(benchmark_results.keys())
        metrics = list(next(iter(benchmark_results.values())).keys())

        fig = go.Figure()

        for metric in metrics:
            values = [benchmark_results[ratio][metric] for ratio in compression_ratios]
            fig.add_trace(
                go.Scatter(
                    x=compression_ratios,
                    y=values,
                    mode="lines+markers",
                    name=metric.upper(),
                    line=dict(width=3),
                    marker=dict(size=8),
                )
            )

        fig.update_layout(
            title="Benchmark Performance vs Compression Ratio",
            xaxis_title="Compression Ratio",
            yaxis_title="Score",
            hovermode="x unified",
            width=800,
            height=500,
        )

        return fig

    def create_compression_report(
        self, compression_result, benchmark_results: Dict = None
    ) -> str:
        """압축 결과 리포트 생성"""
        report = f"""
# EHPC 압축 결과 리포트

## 📊 압축 통계
- **원본 길이**: {len(compression_result.original_tokens)} 토큰
- **압축 길이**: {len(compression_result.compressed_tokens)} 토큰  
- **압축률**: {compression_result.compression_ratio:.1%}
- **제거된 토큰**: {len(compression_result.original_tokens) - len(compression_result.compressed_tokens)} 개

## 🎯 사용된 Evaluator Heads
"""

        for head in compression_result.evaluator_heads:
            report += f"- Layer {head.layer}, Head {head.head}: 신뢰도 {head.confidence_score:.3f}\n"

        if benchmark_results:
            report += "\n## 📈 벤치마크 성능\n"
            for ratio, metrics in benchmark_results.items():
                report += f"\n### 압축률 {ratio}\n"
                for metric, score in metrics.items():
                    report += f"- {metric.upper()}: {score:.3f}\n"

        report += """
## 🔍 중요도 상위 토큰들
"""

        # 상위 10개 중요 토큰 표시
        top_indices = np.argsort(compression_result.token_scores)[-10:][::-1]
        for i, idx in enumerate(top_indices):
            token = compression_result.original_tokens[idx]
            score = compression_result.token_scores[idx]
            kept = "✅" if idx in compression_result.selected_indices else "❌"
            report += f"{i+1}. {token} (점수: {score:.3f}) {kept}\n"

        return report
