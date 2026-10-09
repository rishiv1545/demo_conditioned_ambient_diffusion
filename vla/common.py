"""SmolVLA helpers: device, datasets, policy/processor construction, lightweight checkpoints."""
import glob
import json
import os
import re

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

BASE = "lerobot/smolvla_base"


def pick_device(name="auto"):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def base_config():
    """SmolVLA-base config. Importing the smolvla config module registers the "smolvla" choice with draccus."""
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    return PreTrainedConfig.from_pretrained(BASE)


def open_datasets(roots, chunk_size, fps=10):
    """LeRobotDatasets with action chunks of length chunk_size. Returns (list of datasets, aggregated stats)."""
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dts = {"action": [i / fps for i in range(chunk_size)]}
    dss = [LeRobotDataset(f"local/{os.path.basename(os.path.normpath(r))}", root=r, delta_timestamps=dts)
           for r in roots]
    stats = aggregate_stats([d.meta.stats for d in dss]) if len(dss) > 1 else dss[0].meta.stats
    return dss, stats


def resolve_dtype(device, dtype="auto"):
    """SmolVLA's default (bf16 backbone) where bf16 is supported (MPS, Ampere+ CUDA); otherwise everything in fp32
    (e.g. on a T4, which has no bf16). A full fp16 cast is not offered: the backbone was trained in bf16 (fp16
    can overflow), and fp32 inputs such as the state would then hit fp16 layers."""
    if dtype != "auto":
        return {"fp32": torch.float32, "keep": None}[dtype]
    if device == "cuda" and not torch.cuda.is_bf16_supported():
        return torch.float32
    if device == "cpu":
        return torch.float32
    return None  # keep SmolVLA's default (bf16 backbone, fp32 elsewhere)


def build_policy(features, stats, device, n_action_steps=None, dtype="auto", unfreeze_vlm=False):
    """SmolVLA-base weights with our input/output features (4-D state, 4-D action, our cameras).
    unfreeze_vlm: also train the VLM's language-model layers (train_expert_only=False). SigLIP and the connector stay
    frozen, so the vision-feature cache (post-connector) stays valid; grounding "red" to the red cube happens in
    the language layers, where image and instruction tokens attend to each other."""
    from lerobot.configs.types import FeatureType
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
    cfg = base_config()
    cfg.pretrained_path = BASE
    cfg.device = device
    pf = dataset_to_policy_features(features)
    cfg.output_features = {k: f for k, f in pf.items() if f.type is FeatureType.ACTION}
    cfg.input_features = {k: f for k, f in pf.items() if f.type is not FeatureType.ACTION}
    if n_action_steps is not None:
        cfg.n_action_steps = n_action_steps
    cfg.train_expert_only = not unfreeze_vlm
    policy = SmolVLAPolicy.from_pretrained(BASE, config=cfg)
    if unfreeze_vlm:
        vlm = policy.model.vlm_with_expert.get_vlm_model()
        for prm in list(vlm.vision_model.parameters()) + list(vlm.connector.parameters()):
            prm.requires_grad = False
    dt = resolve_dtype(device, dtype)
    if dt is not None:
        policy = policy.to(dt)
    policy = policy.to(device)
    pre, post = make_smolvla_pre_post_processors(cfg, dataset_stats=stats)
    return policy, pre, post, cfg


def training_side_info(rc):
    """episodes.json of the run's (first) training dataset: held-out split and env version. Must exist: without it
    eval/probe would silently use the v1 env (old gripper orientation/start state)."""
    path = os.path.join(rc["args"]["data"][0], "episodes.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: download/export the run's training dataset first "
                                f"(on Colab: run the dataset-download cell)")
    with open(path) as f:
        return json.load(f)


def image_spec(features):
    """(camera names, image size) from dataset/policy features ("observation.images.<cam>", shape [H, W, C])."""
    keys = sorted(k for k in features if k.startswith("observation.images."))
    return [k.rsplit(".", 1)[1] for k in keys], int(features[keys[0]]["shape"][0])


def trainable_state(policy):
    return {n: p.detach().cpu() for n, p in policy.named_parameters() if p.requires_grad}


def save_checkpoint(run_dir, step, policy, opt=None, sched=None, extra=None):
    d = os.path.join(run_dir, f"step_{step:06d}")
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, "trainable.pt.tmp")
    torch.save(trainable_state(policy), tmp)
    os.replace(tmp, os.path.join(d, "trainable.pt"))  # atomic: a disconnect never leaves a half-written file
    if opt is not None:
        st = {"step": step, "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
              "rng_torch": torch.get_rng_state(), "rng_np": np.random.get_state(), **(extra or {})}
        torch.save(st, os.path.join(d, "train_state.pt.tmp"))
        os.replace(os.path.join(d, "train_state.pt.tmp"), os.path.join(d, "train_state.pt"))
        # optimizer state is only needed to resume: keep it for the newest checkpoint only (it's 2x the weights)
        for old_step, old in list_checkpoints(run_dir):
            if old_step < step and os.path.exists(os.path.join(old, "train_state.pt")):
                os.remove(os.path.join(old, "train_state.pt"))
    return d


def list_checkpoints(run_dir):
    out = []
    for d in glob.glob(os.path.join(run_dir, "step_*")):
        m = re.search(r"step_(\d+)$", d)
        if m and os.path.exists(os.path.join(d, "trainable.pt")):
            out.append((int(m.group(1)), d))
    return sorted(out)


def load_trainable(policy, ckpt_dir):
    sd = torch.load(os.path.join(ckpt_dir, "trainable.pt"), map_location="cpu")
    missing = [n for n, p in policy.named_parameters() if p.requires_grad and n not in sd]
    if missing:
        raise RuntimeError(f"checkpoint lacks {len(missing)} trainable params, e.g. {missing[:3]}")
    policy.load_state_dict(sd, strict=False)


def save_run_config(run_dir, cfg_dict, features, stats):
    def tolist(x):
        if isinstance(x, dict):
            return {k: tolist(v) for k, v in x.items()}
        return np.asarray(x).tolist()
    with open(os.path.join(run_dir, "run.json"), "w") as f:
        json.dump({"args": cfg_dict, "features": features, "stats": tolist(stats)}, f, indent=1, default=str)


def load_run_config(run_dir):
    with open(os.path.join(run_dir, "run.json")) as f:
        r = json.load(f)
    r["stats"] = {k: {s: np.asarray(v, dtype=np.float32) for s, v in d.items()} for k, d in r["stats"].items()}
    for k, ft in r["features"].items():
        ft["shape"] = tuple(ft["shape"])
    return r
