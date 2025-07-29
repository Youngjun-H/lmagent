#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 토크나이저 어휘 확장을 위한 토큰 후보 분석 스크립트
전체 데이터셋을 처리하여 새로운 토큰 후보를 식별합니다.

핵심 기능:
- 스코어 기반 토큰 후보 선정: 빈도 × (토큰 수 - 1)
- 2개 이상의 토큰으로 쪼개지는 단어만 후보로 간주
- 빈도와 토큰 효율성을 모두 고려한 우선순위 계산
"""

import gc
import json
import os
import re
from collections import Counter

from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoTokenizer


def load_environment():
    """환경 변수 및 라이브러리 로드"""
    print("환경 설정을 로드하는 중...")
    load_dotenv()

    # OpenAI API 키 확인 (선택사항)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("OpenAI API 키가 설정되었습니다.")
    else:
        print("OpenAI API 키가 설정되지 않았습니다.")


def load_tokenizer():
    """토크나이저 로드"""
    print("토크나이저를 로드하는 중...")
    tokenizer = AutoTokenizer.from_pretrained(
        "skt/A.X-4.0-Light",
        cache_dir="/purestorage/AILAB/AI_2/yjhwang/work/cache/hf",
        trust_remote_code=True,
    )
    print(f"토크나이저 어휘 크기: {len(tokenizer.get_vocab())}")
    return tokenizer


def load_dataset_data():
    """데이터셋 로드"""
    print("한국어 웹텍스트 데이터셋을 로드하는 중...")
    try:
        ds = load_dataset(
            "HAERAE-HUB/KOREAN-WEBTEXT",
            cache_dir="/purestorage/AILAB/AI_2/yjhwang/work/cache/hf",
            trust_remote_code=True,
        )
        print(f"훈련 데이터 샘플 수: {len(ds['train'])}")
        return ds
    except Exception as e:
        print(f"데이터셋 로드 중 오류 발생: {e}")
        return None


def process_dataset_in_batches(dataset, batch_size=1000):
    """배치 단위로 데이터셋을 처리하여 단어 빈도 계산"""
    print("전체 데이터셋에서 단어 빈도를 계산하는 중...")
    word_counts = Counter()

    total_batches = (len(dataset["train"]) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(dataset["train"]), batch_size),
        total=total_batches,
        desc="배치 처리 중",
    ):
        try:
            # 배치 데이터 추출
            batch_texts = dataset["train"][i : i + batch_size]["text"]

            # 텍스트를 하나로 합치고 단어 추출
            combined_text = " ".join(batch_texts)
            words = re.findall(r"\w+", combined_text.lower())

            # 빈도 업데이트
            word_counts.update(words)

            # 메모리 정리
            del batch_texts, combined_text, words
            gc.collect()

        except Exception as e:
            print(f"배치 {i//batch_size + 1} 처리 중 오류: {e}")
            continue

    print(f"총 {len(word_counts)}개의 고유 단어를 발견했습니다.")
    return word_counts


def print_word_statistics(word_counts):
    """단어 빈도 통계 출력"""
    print("\n=== 단어 빈도 통계 ===")
    print(f"빈도 1-4: {sum(1 for count in word_counts.values() if 1 <= count <= 4)}")
    print(f"빈도 5-9: {sum(1 for count in word_counts.values() if 5 <= count <= 9)}")
    print(
        f"빈도 10-99: {sum(1 for count in word_counts.values() if 10 <= count <= 99)}"
    )
    print(
        f"빈도 100-999: {sum(1 for count in word_counts.values() if 100 <= count <= 999)}"
    )
    print(f"빈도 1000+: {sum(1 for count in word_counts.values() if count >= 1000)}")

    print("\n=== 가장 빈도가 높은 단어들 (상위 20개) ===")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")


def find_token_candidates(word_counts, tokenizer, min_frequency=10, min_token_length=2):
    """토큰 후보 식별 - 스코어 기반 선정"""
    print(
        f"\n토큰 후보를 식별하는 중... (최소 빈도: {min_frequency}, 최소 토큰 길이: {min_token_length})"
    )

    candidate_tokens = {}
    candidate_count = 0

    for word, count in tqdm(word_counts.items(), desc="토큰 분석 중"):
        if count < min_frequency:
            continue

        candidate_count += 1
        try:
            # 공백을 추가하여 단어 단위로 토큰화되도록 유도
            tokens = tokenizer.encode(" " + word, add_special_tokens=False)

            # 2개 이상의 토큰으로 쪼개지는 단어만 후보로 간주
            if len(tokens) >= min_token_length:
                # 스코어 계산: 빈도 * (토큰 수 - 1)
                score = count * (len(tokens) - 1)
                candidate_tokens[word] = {
                    "frequency": count,
                    "token_count": len(tokens),
                    "score": score,
                    "tokens": tokens,
                }
        except Exception as e:
            print(f"단어 '{word}' 토큰화 중 오류: {e}")
            continue

    print(f"분석된 후보 단어 수: {candidate_count}")
    print(f"새롭게 추가할 토큰 후보 수: {len(candidate_tokens)}")

    return candidate_tokens, candidate_count


def print_candidate_details(candidate_tokens, tokenizer, top_n=20):
    """토큰 후보들의 상세 정보 출력 - 스코어 순"""
    print(f"\n=== 토큰 후보들의 상세 정보 (상위 {top_n}개) ===")

    # 스코어 기준으로 내림차순 정렬
    sorted_candidates = sorted(
        candidate_tokens.items(), key=lambda item: item[1]["score"], reverse=True
    )

    for i, (word, info) in enumerate(sorted_candidates[:top_n]):
        tokens = info["tokens"]
        decoded = tokenizer.decode(tokens)
        print(f"\n{i+1}. 단어: '{word}'")
        print(f"   빈도: {info['frequency']}")
        print(f"   토큰 수: {info['token_count']}")
        print(f"   스코어: {info['score']} (빈도 × (토큰수-1))")
        print(f"   토큰 ID: {tokens}")
        print(f"   디코딩: '{decoded}'")


def calculate_token_score(frequency, token_count):
    """토큰 스코어 계산: 빈도 × (토큰 수 - 1)"""
    return frequency * (token_count - 1)


def get_top_candidates(candidate_tokens, top_n=10):
    """상위 N개 토큰 후보 반환"""
    sorted_candidates = sorted(
        candidate_tokens.items(), key=lambda item: item[1]["score"], reverse=True
    )
    return [word for word, info in sorted_candidates[:top_n]]


def save_results(
    candidate_tokens,
    tokenizer,
    candidate_count,
    min_frequency,
    min_token_length,
    output_dir="results",
):
    """결과를 JSON과 TXT 파일로 저장 - 스코어 기반"""
    print(f"\n결과를 저장하는 중... (출력 디렉토리: {output_dir})")

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # 스코어 기준으로 내림차순 정렬
    sorted_candidates = sorted(
        candidate_tokens.items(), key=lambda item: item[1]["score"], reverse=True
    )

    # JSON 결과 구성
    results = {
        "total_candidates_analyzed": candidate_count,
        "new_token_candidates": len(candidate_tokens),
        "min_frequency_threshold": min_frequency,
        "min_token_length_threshold": min_token_length,
        "scoring_method": "frequency * (token_count - 1)",
        "candidates_with_details": [],
    }

    for word, info in sorted_candidates:
        tokens = info["tokens"]
        decoded = tokenizer.decode(tokens)

        results["candidates_with_details"].append(
            {
                "word": word,
                "frequency": info["frequency"],
                "token_count": info["token_count"],
                "score": info["score"],
                "token_ids": tokens,
                "decoded": decoded,
            }
        )

    # JSON 파일로 저장
    json_path = os.path.join(output_dir, "token_candidates_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # TXT 파일로 저장 (스코어 순)
    txt_path = os.path.join(output_dir, "new_token_candidates_by_score.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("# 스코어 순 정렬된 새로운 토큰 후보 목록\n")
        f.write(f"# 총 {len(candidate_tokens)}개의 토큰 후보\n")
        f.write(f"# 최소 빈도: {min_frequency}, 최소 토큰 길이: {min_token_length}\n")
        f.write("# 스코어 계산: 빈도 * (토큰 수 - 1)\n")
        f.write("# 형식: 단어 (빈도, 토큰 수, 스코어)\n\n")

        for word, info in sorted_candidates:
            f.write(
                f"{word} (빈도: {info['frequency']}, 토큰: {info['token_count']}, 스코어: {info['score']})\n"
            )

    # 단어만 저장하는 TXT 파일
    words_only_path = os.path.join(output_dir, "new_token_candidates_words_only.txt")
    with open(words_only_path, "w", encoding="utf-8") as f:
        f.write("# 스코어 순 정렬된 토큰 후보 단어 목록\n")
        f.write(f"# 총 {len(candidate_tokens)}개의 토큰 후보\n\n")

        for word, info in sorted_candidates:
            f.write(f"{word}\n")

    print(f"결과가 저장되었습니다:")
    print(f"  - JSON: {json_path}")
    print(f"  - TXT (스코어 순): {txt_path}")
    print(f"  - TXT (단어만): {words_only_path}")

    return json_path, txt_path, words_only_path


def main():
    """메인 함수"""
    print("=== 한국어 토크나이저 어휘 확장 분석 시작 ===\n")

    # 환경 설정
    load_environment()

    # 토크나이저 로드
    tokenizer = load_tokenizer()

    # 데이터셋 로드
    dataset = load_dataset_data()
    if dataset is None:
        print("데이터셋 로드에 실패했습니다. 프로그램을 종료합니다.")
        return

    # 전체 데이터셋 처리
    word_counts = process_dataset_in_batches(dataset)

    # 통계 출력
    print_word_statistics(word_counts)

    # 토큰 후보 식별
    min_frequency = 10  # 최소 빈도 임계값
    min_token_length = 2  # 최소 토큰 길이 임계값 (2개 이상 토큰으로 쪼개지는 단어)

    candidate_tokens, candidate_count = find_token_candidates(
        word_counts, tokenizer, min_frequency, min_token_length
    )

    # 상세 정보 출력 (스코어 순으로 상위 20개)
    if len(candidate_tokens) > 0:
        print_candidate_details(candidate_tokens, tokenizer, top_n=20)
    else:
        print("\n조건을 만족하는 토큰 후보가 없습니다.")

    # 결과 저장
    save_results(
        candidate_tokens,
        tokenizer,
        candidate_count,
        min_frequency,
        min_token_length,
    )

    print(f"\n=== 분석 완료 ===")
    print(f"총 {len(candidate_tokens)}개의 토큰 후보를 발견했습니다.")

    # 최종 추가할 토큰 목록 (상위 N개 선택 예시)
    if len(candidate_tokens) > 0:
        final_tokens = get_top_candidates(candidate_tokens, top_n=10)
        print(f"추천 추가 토큰 (상위 10개): {final_tokens}")

        # 스코어 계산 예시 출력
        print("\n=== 스코어 계산 예시 ===")
        for word in final_tokens[:3]:  # 상위 3개만 예시로 출력
            info = candidate_tokens[word]
            calculated_score = calculate_token_score(
                info["frequency"], info["token_count"]
            )
            print(
                f"'{word}': 빈도 {info['frequency']} × (토큰수 {info['token_count']} - 1) = {calculated_score}"
            )


if __name__ == "__main__":
    main()
