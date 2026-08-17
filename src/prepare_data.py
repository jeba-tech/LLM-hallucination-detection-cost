"""Download TruthfulQA and write a subset to data/truthfulqa_subset.csv.

Run once: python -m src.prepare_data
"""
import pandas as pd
from datasets import load_dataset

import config


def main():
    # Fully-qualified repo id: the bare "truthful_qa" alias no longer resolves
    # under current huggingface_hub, which requires namespace/name.
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    df = ds.to_pandas()

    # Samples are *nested*: the first PILOT_SIZE rows are exactly the pilot
    # sample, and larger sizes append to it. results/raw_runs.csv keys rows by
    # position in this file, so a plain resample at a new size would silently
    # repoint every existing question_id at a different question and invalidate
    # both the logged results and the resume logic.
    size = min(config.DATASET_SUBSET_SIZE, len(df))
    pilot = df.sample(n=min(config.PILOT_SIZE, size), random_state=config.RANDOM_SEED)
    if size > len(pilot):
        extra = df.drop(pilot.index).sample(
            n=size - len(pilot), random_state=config.RANDOM_SEED
        )
        df = pd.concat([pilot, extra])
    else:
        df = pilot
    df = df.reset_index(drop=True)

    out = pd.DataFrame(
        {
            "question": df["question"],
            "correct_answer": df["best_answer"],
            "incorrect_answer": df["incorrect_answers"].apply(
                lambda xs: xs[0] if len(xs) else ""
            ),
        }
    )

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.DATASET_PATH, index=False)
    print(f"Wrote {len(out)} rows to {config.DATASET_PATH}")


if __name__ == "__main__":
    main()
