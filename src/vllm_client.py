import time
import os
from typing import Dict, Any


def run_vllm_benchmark(
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    prompt: str = "Explain how machine learning model serving works in MLOps."
) -> Dict[str, Any]:
    """
    Connects to vLLM OpenAI-compatible server, enables streaming mode,
    and measures Time To First Token (TTFT) and generation throughput (tokens/sec).
    """
    print(f"[vLLM Client] Connecting to {base_url} using model '{model_name}'...")
    print(f"[vLLM Client] Prompt: '{prompt}'")

    try:
        from openai import OpenAI

        client = OpenAI(base_url=base_url, api_key="EMPTY")

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0
        full_text = []

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.7,
            max_tokens=200,
        )

        for chunk in response:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                content = chunk.choices[0].delta.content
                full_text.append(content)
                token_count += 1

        end_time = time.perf_counter()

        ttft_ms = round(((first_token_time or end_time) - start_time) * 1000, 2)
        total_duration = end_time - start_time
        gen_duration = end_time - (first_token_time or start_time)
        throughput = round(token_count / gen_duration, 2) if gen_duration > 0 else 0.0

        print("\n" + "="*50)
        print("vLLM SERVING BENCHMARK METRICS")
        print("="*50)
        print(f"Time To First Token (TTFT) : {ttft_ms} ms")
        print(f"Total Generated Tokens     : {token_count}")
        print(f"Generation Throughput      : {throughput} tokens/sec")
        print(f"Total Request Duration     : {round(total_duration, 3)} sec")
        print("="*50 + "\n")

        return {
            "status": "success",
            "ttft_ms": ttft_ms,
            "token_count": token_count,
            "throughput_tokens_per_sec": throughput,
            "total_duration_sec": round(total_duration, 3)
        }

    except Exception as e:
        print(f"[vLLM Client] Live vLLM server unavailable ({e}). Demonstrating mock benchmark mode...")
        # Fallback simulation for environments without GPU / active vLLM instance
        t0 = time.perf_counter()
        time.sleep(0.045)  # Simulate 45ms TTFT
        t_first = time.perf_counter()

        simulated_tokens = 150
        time.sleep(1.2)  # Simulate generation duration
        t_end = time.perf_counter()

        ttft_ms = round((t_first - t0) * 1000, 2)
        throughput = round(simulated_tokens / (t_end - t_first), 2)

        print("\n" + "="*50)
        print("vLLM SERVING MOCK BENCHMARK METRICS")
        print("="*50)
        print(f"Time To First Token (TTFT) : {ttft_ms} ms")
        print(f"Total Generated Tokens     : {simulated_tokens}")
        print(f"Generation Throughput      : {throughput} tokens/sec")
        print("="*50 + "\n")

        return {
            "status": "mock_success",
            "ttft_ms": ttft_ms,
            "token_count": simulated_tokens,
            "throughput_tokens_per_sec": throughput
        }


if __name__ == "__main__":
    run_vllm_benchmark()
