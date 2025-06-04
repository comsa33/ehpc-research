"""
논문의 실험을 재현하는 업그레이드된 스크립트
최신 모델들과 한국어 지원을 포함한 종합적인 EHPC 성능 검증
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import torch
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.evaluator_heads import ModelConfig, get_model_info
from core.prompt_compressor import EHPCCompressor, create_compressor
from evaluation.benchmarks import EHPCBenchmark
from visualization.attention_viz import AttentionVisualizer

# Rich 콘솔 설정
console = Console()


# 로깅 설정
def setup_logging(log_file: str):
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


class EHPCExperimentRunner:
    """EHPC 실험 실행기 (업그레이드 버전)"""

    def __init__(self, output_dir: str = "experiment_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 로깅 설정
        log_file = self.output_dir / f"experiment_{self.timestamp}.log"
        setup_logging(str(log_file))

        # 실험 설정
        self.compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.test_prompts = self._get_test_prompts()

        console.print(
            Panel.fit(
                f"🧪 EHPC 실험 실행기 v2.0\n"
                f"📅 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"📁 결과 디렉토리: {self.output_dir}",
                title="실험 초기화",
                border_style="blue",
            )
        )

    def _get_test_prompts(self) -> Dict[str, List[str]]:
        """테스트 프롬프트 생성 (다국어 지원)"""
        return {
            "english": [
                "Explain the concept of machine learning in simple terms.",
                "What are the key differences between supervised and unsupervised learning?",
                "Describe the attention mechanism in transformer models and its importance.",
                "How does gradient descent work in neural network training?",
                "What are the advantages and disadvantages of deep learning?",
                "Compare different optimization algorithms used in machine learning.",
                "Explain the concept of overfitting and how to prevent it.",
                "What is the difference between classification and regression tasks?",
            ],
            "korean": [
                "머신러닝의 개념을 간단한 용어로 설명해주세요.",
                "지도학습과 비지도학습의 주요 차이점은 무엇인가요?",
                "트랜스포머 모델의 어텐션 메커니즘과 그 중요성을 설명해주세요.",
                "신경망 훈련에서 경사하강법은 어떻게 작동하나요?",
                "딥러닝의 장점과 단점은 무엇인가요?",
                "기계학습에서 사용되는 다양한 최적화 알고리즘을 비교해주세요.",
                "과적합의 개념과 이를 방지하는 방법을 설명해주세요.",
                "분류 작업과 회귀 작업의 차이점은 무엇인가요?",
            ],
            "technical": [
                "The transformer architecture revolutionized NLP through self-attention mechanisms, enabling parallel processing and capturing long-range dependencies in sequences.",
                "Large language models utilize billions of parameters to understand and generate human-like text across diverse domains and languages.",
                "Prompt engineering involves crafting input instructions to optimize model performance for specific tasks without fine-tuning.",
                "Attention heads in transformers specialize in different linguistic phenomena, from syntax parsing to semantic understanding.",
                "BERT's bidirectional encoder enables better context understanding compared to traditional left-to-right language models.",
            ],
        }

    def get_test_models(self, model_filter: Optional[str] = None) -> List[str]:
        """테스트할 모델 목록 생성"""
        all_models = {
            # Tier 1: 최고 성능 (충분한 GPU 메모리 필요)
            "tier1": [
                "meta-llama/Llama-3.2-3B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "beomi/KoAlpaca-Polyglot-5.8B",
            ],
            # Tier 2: 균형형
            "tier2": [
                "microsoft/Phi-3.5-mini-instruct",
                "google/gemma-2-2b",
                "Qwen/Qwen2.5-1.5B-Instruct",
            ],
            # Tier 3: 경량형 (기본 테스트용)
            "tier3": ["microsoft/DialoGPT-medium", "google/gemma-2-2b"],
        }

        if model_filter:
            if model_filter in all_models:
                return all_models[model_filter]
            elif model_filter in ModelConfig.SUPPORTED_MODELS:
                return [model_filter]
            else:
                console.print(f"[red]알 수 없는 모델: {model_filter}[/red]")
                return []

        # 하드웨어 기반 자동 선택
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 16:
                return all_models["tier1"]
            elif gpu_memory >= 8:
                return all_models["tier2"][:2] + [all_models["tier1"][1]]  # Qwen 포함
            else:
                return all_models["tier3"]
        else:
            # CPU 환경
            return ["microsoft/DialoGPT-medium"]

    def run_comprehensive_experiments(
        self,
        models: Optional[List[str]] = None,
        include_qa: bool = True,
        include_generation: bool = True,
        include_compression: bool = True,
        num_qa_samples: int = 30,
    ) -> Dict:
        """포괄적인 EHPC 실험 실행"""

        if models is None:
            models = self.get_test_models()

        console.print(f"🎯 테스트 모델: {', '.join(models)}")

        all_results = {
            "experiment_info": {
                "timestamp": self.timestamp,
                "models_tested": models,
                "compression_ratios": self.compression_ratios,
                "qa_samples": num_qa_samples if include_qa else 0,
                "generation_samples": (
                    len(self.test_prompts["english"]) if include_generation else 0
                ),
                "hardware": self._get_hardware_info(),
            },
            "results": {},
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            overall_task = progress.add_task("전체 실험 진행", total=len(models))

            for model_idx, model_name in enumerate(models):
                model_task = progress.add_task(
                    f"모델 {model_name.split('/')[-1]} 테스트", total=100
                )

                try:
                    console.print(f"\n{'='*80}")
                    console.print(
                        f"🧪 모델 테스트: [bold blue]{model_name}[/bold blue]"
                    )
                    console.print(f"{'='*80}")

                    # 모델 로딩
                    progress.update(
                        model_task, advance=10, description="모델 로딩 중..."
                    )
                    compressor = self._load_model_safely(model_name)

                    if compressor is None:
                        console.print(
                            f"[red]❌ 모델 {model_name} 로딩 실패, 스킵[/red]"
                        )
                        progress.update(overall_task, advance=1)
                        continue

                    model_results = {
                        "model_name": model_name,
                        "model_info": compressor.get_compression_stats(),
                        "experiments": {},
                    }

                    # 1. 압축 성능 테스트
                    if include_compression:
                        progress.update(
                            model_task, advance=20, description="압축 성능 테스트..."
                        )
                        compression_results = self._run_compression_tests(compressor)
                        model_results["experiments"][
                            "compression"
                        ] = compression_results

                    # 2. QA 벤치마크
                    if include_qa:
                        progress.update(
                            model_task, advance=30, description="QA 벤치마크..."
                        )
                        qa_results = self._run_qa_benchmark(compressor, num_qa_samples)
                        model_results["experiments"]["qa"] = qa_results

                    # 3. 생성 테스트
                    if include_generation:
                        progress.update(
                            model_task, advance=30, description="생성 테스트..."
                        )
                        generation_results = self._run_generation_tests(compressor)
                        model_results["experiments"]["generation"] = generation_results

                    progress.update(model_task, advance=10, description="결과 저장...")
                    all_results["results"][model_name] = model_results

                    # 중간 결과 저장
                    self._save_intermediate_results(model_name, model_results)

                    console.print(f"✅ 모델 {model_name} 테스트 완료")

                except Exception as e:
                    console.print(f"[red]❌ 모델 {model_name} 테스트 실패: {e}[/red]")
                    logging.error(f"모델 {model_name} 테스트 실패: {e}")
                    continue

                finally:
                    progress.update(overall_task, advance=1)
                    progress.remove_task(model_task)

                    # 메모리 정리
                    self._cleanup_memory()

        # 최종 결과 저장
        final_results_file = (
            self.output_dir / f"comprehensive_results_{self.timestamp}.json"
        )
        with open(final_results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 결과 요약 출력
        self._print_experiment_summary(all_results)

        # 시각화 리포트 생성
        self._create_visualization_report(final_results_file)

        return all_results

    def _load_model_safely(self, model_name: str) -> Optional[EHPCCompressor]:
        """안전한 모델 로딩"""
        try:
            compressor = create_compressor(model_name, auto_initialize=False)

            # 초기화 수행 (모델 크기에 따라 조정)
            model_info = get_model_info(model_name)
            params = model_info.get("params", "2B")

            if "5B" in params or "8B" in params:
                max_layers, heads_per_layer = 2, 1  # 큰 모델은 제한적으로
            elif "3B" in params:
                max_layers, heads_per_layer = 3, 2
            else:
                max_layers, heads_per_layer = 3, 2

            compressor.initialize(
                max_layers=max_layers, heads_per_layer=heads_per_layer
            )

            return compressor

        except Exception as e:
            logging.error(f"모델 {model_name} 로딩 실패: {e}")
            return None

    def _run_compression_tests(self, compressor: EHPCCompressor) -> Dict:
        """압축 성능 테스트"""
        test_texts = []

        # 다양한 언어의 텍스트 준비
        for lang_key, prompts in self.test_prompts.items():
            test_texts.extend(prompts[:3])  # 각 언어에서 3개씩

        results = compressor.benchmark_compression(
            test_texts=test_texts, compression_ratios=[0.2, 0.3, 0.5, 0.7]
        )

        return results

    def _run_qa_benchmark(self, compressor: EHPCCompressor, num_samples: int) -> Dict:
        """QA 벤치마크 실행"""
        try:
            benchmark = EHPCBenchmark(compressor)
            results = benchmark.run_qa_benchmark(
                num_samples=num_samples, compression_ratios=[0.3, 0.5, 0.7]
            )
            return results
        except Exception as e:
            logging.warning(f"QA 벤치마크 실패: {e}")
            return {"error": str(e)}

    def _run_generation_tests(self, compressor: EHPCCompressor) -> Dict:
        """생성 테스트 실행"""
        results = {"english_tests": [], "korean_tests": [], "technical_tests": []}

        # 각 언어별 테스트
        for lang_key, prompts in self.test_prompts.items():
            lang_results = []

            for i, prompt in enumerate(prompts[:3]):  # 각 언어에서 3개씩 테스트
                try:
                    result = compressor.compress_and_generate(
                        prompt,
                        compression_ratio=0.3,
                        max_new_tokens=80,
                        temperature=0.7,
                    )

                    lang_results.append(
                        {
                            "prompt_id": i,
                            "original_prompt": prompt[:100] + "...",  # 처음 100자만
                            "compression_ratio": result["compression_ratio"],
                            "original_length": result["original_length"],
                            "compressed_length": result["compressed_length"],
                            "generation_time_original": result.get(
                                "generation_time_original", 0
                            ),
                            "generation_time_compressed": result.get(
                                "generation_time_compressed", 0
                            ),
                            "speedup": result.get("generation_time_original", 0)
                            / max(result.get("generation_time_compressed", 1), 0.001),
                        }
                    )

                except Exception as e:
                    logging.warning(
                        f"생성 테스트 실패 (언어: {lang_key}, 프롬프트 {i}): {e}"
                    )
                    continue

            results[f"{lang_key}_tests"] = lang_results

        return results

    def _get_hardware_info(self) -> Dict:
        """하드웨어 정보 수집"""
        info = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            info.update(
                {
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0),
                    "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory
                    / (1024**3),
                }
            )

        return info

    def _save_intermediate_results(self, model_name: str, results: Dict):
        """중간 결과 저장"""
        safe_model_name = model_name.replace("/", "_")
        intermediate_file = (
            self.output_dir / f"results_{safe_model_name}_{self.timestamp}.json"
        )

        with open(intermediate_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    def _cleanup_memory(self):
        """메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import gc

        gc.collect()

    def _print_experiment_summary(self, results: Dict):
        """실험 결과 요약 출력"""
        console.print("\n" + "=" * 80)
        console.print("📊 EHPC 실험 결과 요약", style="bold blue")
        console.print("=" * 80)

        # 전체 통계
        experiment_info = results["experiment_info"]
        console.print(f"🕐 실험 시간: {experiment_info['timestamp']}")
        console.print(f"🎯 테스트 모델: {len(experiment_info['models_tested'])}개")
        console.print(f"📊 압축 비율: {len(experiment_info['compression_ratios'])}개")

        # 모델별 결과 테이블
        table = Table(title="모델별 성능 요약")
        table.add_column("모델", style="cyan")
        table.add_column("파라미터", style="magenta")
        table.add_column("Evaluator Heads", style="green")
        table.add_column("평균 압축률", style="yellow")
        table.add_column("QA 성능", style="blue")
        table.add_column("생성 속도", style="red")

        for model_name, model_results in results["results"].items():
            model_short = model_name.split("/")[-1]

            # 기본 정보
            model_info = model_results.get("model_info", {}).get("model_info", {})
            params = model_info.get("parameters", "Unknown")
            heads_count = model_results.get("model_info", {}).get(
                "num_evaluator_heads", 0
            )

            # 압축 성능
            compression_exp = model_results.get("experiments", {}).get(
                "compression", {}
            )
            avg_compression = "N/A"
            if compression_exp and "0.3" in compression_exp:
                avg_compression = (
                    f"{compression_exp['0.3'].get('avg_actual_ratio', 0):.1%}"
                )

            # QA 성능
            qa_exp = model_results.get("experiments", {}).get("qa", {})
            qa_performance = "N/A"
            if qa_exp and not qa_exp.get("error") and "0.3" in qa_exp:
                f1_score = qa_exp["0.3"].get("f1", 0)
                qa_performance = f"{f1_score:.3f}"

            # 생성 속도
            gen_exp = model_results.get("experiments", {}).get("generation", {})
            gen_speed = "N/A"
            if gen_exp.get("english_tests"):
                avg_speedup = sum(
                    test.get("speedup", 1) for test in gen_exp["english_tests"]
                ) / len(gen_exp["english_tests"])
                gen_speed = f"{avg_speedup:.1f}x"

            table.add_row(
                model_short,
                str(params),
                str(heads_count),
                avg_compression,
                qa_performance,
                gen_speed,
            )

        console.print(table)

        # 추가 인사이트
        console.print("\n💡 주요 인사이트:")

        # 최고 성능 모델 찾기
        best_models = self._find_best_models(results["results"])
        for category, model_name in best_models.items():
            if model_name:
                console.print(f"• {category}: [bold]{model_name.split('/')[-1]}[/bold]")

    def _find_best_models(self, results: Dict) -> Dict[str, Optional[str]]:
        """최고 성능 모델 찾기"""
        best = {
            "최고 QA 성능": None,
            "최고 압축 효율": None,
            "최고 생성 속도": None,
            "한국어 최적": None,
        }

        best_qa_score = 0
        best_compression_ratio = 0
        best_gen_speed = 0

        for model_name, model_results in results.items():
            experiments = model_results.get("experiments", {})

            # QA 성능
            qa_exp = experiments.get("qa", {})
            if qa_exp and not qa_exp.get("error") and "0.3" in qa_exp:
                f1_score = qa_exp["0.3"].get("f1", 0)
                if f1_score > best_qa_score:
                    best_qa_score = f1_score
                    best["최고 QA 성능"] = model_name

            # 압축 효율
            compression_exp = experiments.get("compression", {})
            if compression_exp and "0.3" in compression_exp:
                ratio = compression_exp["0.3"].get("avg_actual_ratio", 0)
                if ratio > best_compression_ratio:
                    best_compression_ratio = ratio
                    best["최고 압축 효율"] = model_name

            # 생성 속도
            gen_exp = experiments.get("generation", {})
            if gen_exp.get("english_tests"):
                avg_speedup = sum(
                    test.get("speedup", 1) for test in gen_exp["english_tests"]
                ) / len(gen_exp["english_tests"])
                if avg_speedup > best_gen_speed:
                    best_gen_speed = avg_speedup
                    best["최고 생성 속도"] = model_name

            # 한국어 최적 (KoAlpaca나 Qwen 선호)
            if "koalpaca" in model_name.lower() or "qwen" in model_name.lower():
                best["한국어 최적"] = model_name

        return best

    def _create_visualization_report(self, results_file: Path):
        """시각화 리포트 생성"""
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results = json.load(f)

            visualizer = AttentionVisualizer()

            # 모델별 성능 비교 차트 생성
            import matplotlib.pyplot as plt

            plt.style.use("default")

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle("EHPC 실험 결과 종합 분석", fontsize=16, fontweight="bold")

            # 데이터 준비
            models = []
            qa_scores = []
            compression_ratios = []
            gen_speeds = []
            head_counts = []

            for model_name, model_results in results["results"].items():
                models.append(model_name.split("/")[-1])

                # QA 성능
                qa_exp = model_results.get("experiments", {}).get("qa", {})
                if qa_exp and not qa_exp.get("error") and "0.3" in qa_exp:
                    qa_scores.append(qa_exp["0.3"].get("f1", 0))
                else:
                    qa_scores.append(0)

                # 압축률
                compression_exp = model_results.get("experiments", {}).get(
                    "compression", {}
                )
                if compression_exp and "0.3" in compression_exp:
                    compression_ratios.append(
                        compression_exp["0.3"].get("avg_actual_ratio", 0)
                    )
                else:
                    compression_ratios.append(0)

                # 생성 속도
                gen_exp = model_results.get("experiments", {}).get("generation", {})
                if gen_exp.get("english_tests"):
                    avg_speedup = sum(
                        test.get("speedup", 1) for test in gen_exp["english_tests"]
                    ) / len(gen_exp["english_tests"])
                    gen_speeds.append(avg_speedup)
                else:
                    gen_speeds.append(1)

                # 헤드 수
                head_counts.append(
                    model_results.get("model_info", {}).get("num_evaluator_heads", 0)
                )

            # 1. QA 성능 차트
            if qa_scores and any(qa_scores):
                axes[0, 0].bar(models, qa_scores, color="skyblue", alpha=0.7)
                axes[0, 0].set_title("QA 성능 (F1 Score)")
                axes[0, 0].set_ylabel("F1 Score")
                axes[0, 0].tick_params(axis="x", rotation=45)

            # 2. 압축률 차트
            if compression_ratios and any(compression_ratios):
                axes[0, 1].bar(
                    models, compression_ratios, color="lightcoral", alpha=0.7
                )
                axes[0, 1].set_title("평균 압축률")
                axes[0, 1].set_ylabel("압축률")
                axes[0, 1].tick_params(axis="x", rotation=45)

            # 3. 생성 속도 차트
            if gen_speeds and any(s > 1 for s in gen_speeds):
                axes[1, 0].bar(models, gen_speeds, color="lightgreen", alpha=0.7)
                axes[1, 0].set_title("생성 속도 향상")
                axes[1, 0].set_ylabel("속도 배수")
                axes[1, 0].tick_params(axis="x", rotation=45)

            # 4. Evaluator Heads 수
            if head_counts and any(head_counts):
                axes[1, 1].bar(models, head_counts, color="gold", alpha=0.7)
                axes[1, 1].set_title("Evaluator Heads 수")
                axes[1, 1].set_ylabel("헤드 수")
                axes[1, 1].tick_params(axis="x", rotation=45)

            plt.tight_layout()

            # 파일 저장
            report_file = (
                self.output_dir / f"ehpc_experiment_report_{self.timestamp}.png"
            )
            plt.savefig(report_file, dpi=300, bbox_inches="tight")
            plt.close()

            console.print(f"📊 시각화 리포트 생성: {report_file}")

        except Exception as e:
            console.print(f"[red]⚠️ 시각화 리포트 생성 실패: {e}[/red]")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="EHPC 종합 실험 실행기")
    parser.add_argument("--models", nargs="+", help="테스트할 모델 목록")
    parser.add_argument(
        "--model-tier", choices=["tier1", "tier2", "tier3"], help="모델 티어 선택"
    )
    parser.add_argument(
        "--output-dir", default="experiment_results", help="결과 저장 디렉토리"
    )
    parser.add_argument(
        "--qa-samples", type=int, default=30, help="QA 벤치마크 샘플 수"
    )
    parser.add_argument("--skip-qa", action="store_true", help="QA 벤치마크 스킵")
    parser.add_argument(
        "--skip-generation", action="store_true", help="생성 테스트 스킵"
    )
    parser.add_argument(
        "--skip-compression", action="store_true", help="압축 테스트 스킵"
    )
    parser.add_argument(
        "--quick", action="store_true", help="빠른 테스트 (샘플 수 감소)"
    )

    args = parser.parse_args()

    # 빠른 테스트 설정
    if args.quick:
        args.qa_samples = 10
        console.print("[yellow]⚡ 빠른 테스트 모드 활성화[/yellow]")

    # 실험 실행기 초기화
    runner = EHPCExperimentRunner(args.output_dir)

    # 테스트 모델 결정
    test_models = None
    if args.models:
        test_models = args.models
    elif args.model_tier:
        test_models = runner.get_test_models(args.model_tier)

    # 실험 실행
    try:
        results = runner.run_comprehensive_experiments(
            models=test_models,
            include_qa=not args.skip_qa,
            include_generation=not args.skip_generation,
            include_compression=not args.skip_compression,
            num_qa_samples=args.qa_samples,
        )

        console.print(
            Panel.fit(
                "✅ 모든 실험이 성공적으로 완료되었습니다!\n"
                f"📁 결과는 {runner.output_dir}에 저장되었습니다.",
                title="실험 완료",
                border_style="green",
            )
        )

    except KeyboardInterrupt:
        console.print("[red]❌ 사용자에 의해 실험이 중단되었습니다.[/red]")

    except Exception as e:
        console.print(f"[red]❌ 실험 실행 중 오류 발생: {e}[/red]")
        logging.error(f"실험 실행 오류: {e}")


if __name__ == "__main__":
    main()
