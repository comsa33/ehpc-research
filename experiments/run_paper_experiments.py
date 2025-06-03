"""
논문의 실험을 재현하는 스크립트
다양한 모델과 데이터셋에서 EHPC 성능 검증
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import json
import logging
from datetime import datetime

from core.prompt_compressor import EHPCCompressor
from evaluation.benchmarks import EHPCBenchmark
from visualization.attention_viz import AttentionVisualizer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("experiment_log.txt"), logging.StreamHandler()],
)


def run_comprehensive_experiments():
    """포괄적인 EHPC 실험 실행"""

    # 실험 설정
    models_to_test = [
        "microsoft/DialoGPT-small",
        "microsoft/DialoGPT-medium",
        "gpt2",
    ]

    compression_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for model_name in models_to_test:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing model: {model_name}")
        logging.info(f"{'='*50}")

        try:
            # 압축기 초기화
            compressor = EHPCCompressor(model_name)
            compressor.initialize(max_layers=3, heads_per_layer=2)

            model_results = {
                "model_name": model_name,
                "evaluator_heads": [
                    {
                        "layer": head.layer,
                        "head": head.head,
                        "confidence": head.confidence_score,
                        "selectivity": head.selectivity_score,
                    }
                    for head in compressor.evaluator_heads
                ],
                "qa_results": {},
                "generation_examples": [],
            }

            # QA 벤치마크
            logging.info("Running QA benchmark...")
            benchmark = EHPCBenchmark(compressor)
            qa_results = benchmark.run_qa_benchmark(
                num_samples=50, compression_ratios=compression_ratios
            )
            model_results["qa_results"] = qa_results

            # 생성 예시 테스트
            logging.info("Testing generation examples...")
            test_prompts = [
                "Explain the concept of machine learning in simple terms.",
                "What are the key differences between supervised and unsupervised learning?",
                "Describe the attention mechanism in transformer models.",
                "How does gradient descent work in neural network training?",
                "What are the advantages and disadvantages of deep learning?",
            ]

            for i, prompt in enumerate(test_prompts):
                logging.info(f"Processing example {i+1}/5...")

                try:
                    result = compressor.compress_and_generate(
                        prompt,
                        compression_ratio=0.3,  # 기본 30% 압축
                        max_new_tokens=100,
                    )

                    model_results["generation_examples"].append(
                        {
                            "prompt": prompt,
                            "compression_ratio": result["compression_ratio"],
                            "original_length": result["original_length"],
                            "compressed_length": result["compressed_length"],
                            "original_response": result["original_response"][:200]
                            + "...",
                            "compressed_response": result["compressed_response"][:200]
                            + "...",
                        }
                    )

                except Exception as e:
                    logging.warning(f"Failed to process example {i+1}: {e}")

            all_results[model_name] = model_results

            # 중간 결과 저장
            with open(
                f"results_{model_name.replace('/', '_')}_{timestamp}.json", "w"
            ) as f:
                json.dump(model_results, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logging.error(f"Failed to test model {model_name}: {e}")
            continue

    # 전체 결과 저장
    final_results = {
        "timestamp": timestamp,
        "experiment_config": {
            "models": models_to_test,
            "compression_ratios": compression_ratios,
            "qa_samples": 50,
        },
        "results": all_results,
    }

    with open(f"comprehensive_results_{timestamp}.json", "w") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    # 결과 요약 출력
    print_experiment_summary(final_results)

    return final_results


def print_experiment_summary(results):
    """실험 결과 요약 출력"""
    print(f"\n{'='*60}")
    print("EHPC 실험 결과 요약")
    print(f"{'='*60}")

    for model_name, model_results in results["results"].items():
        print(f"\n📊 모델: {model_name}")
        print(f"   Evaluator Heads: {len(model_results['evaluator_heads'])}개")

        # QA 성능 요약
        qa_results = model_results["qa_results"]
        if qa_results:
            print("   QA 성능 (압축률별):")
            for ratio, metrics in qa_results.items():
                f1_score = metrics.get("f1", 0)
                exact_match = metrics.get("exact_match", 0)
                print(f"     {ratio}: F1={f1_score:.3f}, EM={exact_match:.3f}")

        # 생성 예시 요약
        examples = model_results["generation_examples"]
        if examples:
            avg_compression = sum(ex["compression_ratio"] for ex in examples) / len(
                examples
            )
            print(f"   평균 압축률: {avg_compression:.1%}")

    print("\n💾 상세 결과는 JSON 파일에 저장되었습니다.")


def create_visualization_report(results_file: str):
    """실험 결과 시각화 리포트 생성"""
    with open(results_file, "r") as f:
        results = json.load(f)

    visualizer = AttentionVisualizer()

    # 모델별 성능 비교 차트 생성
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("EHPC 실험 결과 종합", fontsize=16)

    # 1. 압축률별 F1 점수
    ax1 = axes[0, 0]
    for model_name, model_results in results["results"].items():
        qa_results = model_results["qa_results"]
        if qa_results:
            ratios = [float(r) for r in qa_results.keys()]
            f1_scores = [qa_results[str(r)]["f1"] for r in ratios]
            ax1.plot(ratios, f1_scores, marker="o", label=model_name.split("/")[-1])

    ax1.set_title("압축률별 F1 점수")
    ax1.set_xlabel("압축률")
    ax1.set_ylabel("F1 Score")
    ax1.legend()
    ax1.grid(True)

    # 2. Evaluator Heads 분포
    ax2 = axes[0, 1]
    for model_name, model_results in results["results"].items():
        heads = model_results["evaluator_heads"]
        layers = [head["layer"] for head in heads]
        ax2.hist(layers, alpha=0.7, label=model_name.split("/")[-1], bins=range(4))

    ax2.set_title("Evaluator Heads 레이어 분포")
    ax2.set_xlabel("레이어")
    ax2.set_ylabel("헤드 수")
    ax2.legend()

    # 3. 압축 효율성
    ax3 = axes[1, 0]
    model_names = []
    avg_compressions = []

    for model_name, model_results in results["results"].items():
        examples = model_results["generation_examples"]
        if examples:
            avg_compression = sum(ex["compression_ratio"] for ex in examples) / len(
                examples
            )
            model_names.append(model_name.split("/")[-1])
            avg_compressions.append(avg_compression)

    ax3.bar(
        model_names,
        avg_compressions,
        color=["skyblue", "lightcoral", "lightgreen"][: len(model_names)],
    )
    ax3.set_title("모델별 평균 압축률")
    ax3.set_ylabel("압축률")
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)

    # 4. 신뢰도 vs 선택성
    ax4 = axes[1, 1]
    for model_name, model_results in results["results"].items():
        heads = model_results["evaluator_heads"]
        confidences = [head["confidence"] for head in heads]
        selectivities = [head["selectivity"] for head in heads]
        ax4.scatter(
            selectivities, confidences, label=model_name.split("/")[-1], alpha=0.7
        )

    ax4.set_title("Evaluator Heads: 신뢰도 vs 선택성")
    ax4.set_xlabel("선택성 점수")
    ax4.set_ylabel("신뢰도 점수")
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(
        f'ehpc_experiment_report_{results["timestamp"]}.png',
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print(
        f"📊 시각화 리포트가 저장되었습니다: ehpc_experiment_report_{results['timestamp']}.png"
    )


if __name__ == "__main__":
    print("🚀 EHPC 종합 실험 시작...")

    # 실험 실행
    results = run_comprehensive_experiments()

    # 시각화 리포트 생성
    results_file = f"comprehensive_results_{results['timestamp']}.json"
    create_visualization_report(results_file)

    print("✅ 모든 실험이 완료되었습니다!")
