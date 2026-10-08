"""Upload an exported LeRobot dataset to the Hugging Face Hub as a PRIVATE dataset repo.

    python vla/push_hub.py data/lerobot/sim_seen Rishi1545/dcad_sim_seen

The vision cache is not uploaded: it depends on the backbone dtype of the machine that builds it (bf16 on
MPS/Ampere, fp32 on a T4), so each machine builds its own on first use. Download with:
    hf download Rishi1545/dcad_sim_seen --repo-type dataset --local-dir data/lerobot/sim_seen
"""
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("repo_id")
    p.add_argument("--public", action="store_true")
    a = p.parse_args()
    from huggingface_hub import HfApi
    from lerobot.datasets.lerobot_dataset import CODEBASE_VERSION
    api = HfApi()
    api.create_repo(a.repo_id, repo_type="dataset", private=not a.public, exist_ok=True)
    api.upload_folder(repo_id=a.repo_id, repo_type="dataset", folder_path=a.root,
                      ignore_patterns=["vision_cache/*", "*.tmp"], commit_message="dataset export")
    try:
        api.delete_tag(a.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
    except Exception:
        pass
    api.create_tag(a.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")  # LeRobot checks this version tag
    info = api.dataset_info(a.repo_id)
    print(f"pushed {a.root} -> {a.repo_id} (private={info.private}, tag {CODEBASE_VERSION})")


if __name__ == "__main__":
    main()
