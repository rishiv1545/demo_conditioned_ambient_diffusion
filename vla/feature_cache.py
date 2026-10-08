"""Cache SmolVLA's frozen image features, and train from the cache.

With train_expert_only=True (the default) the whole VLM is frozen, including the vision encoder and the
connector, and we use no image augmentation. So `embed_image(prepare(frame))` (resize-with-pad to 512, scale to
[-1, 1], SigLIP, connector) is a fixed function of each frame. We compute it once per frame and store it as fp16
in <dataset>/vision_cache/. Everything after embed_image (the sqrt(dim) scaling, the language/state tokens, the
VLM layers and the action expert) still runs normally.

`CachedChunkDataset` reads states, actions, tasks and sigma_n from the dataset's parquet files plus the cached
features. It never decodes video. `use_cached_features(policy)` patches *this policy instance only*, so that
`prepare_images` returns the cached features and `embed_image` passes them through. Inference/eval always uses
raw images (an unpatched policy); tests/test_feature_cache.py checks that the two paths give the same loss.
"""
import json
import os
import types

import numpy as np
import torch

CACHE_DIR = "vision_cache"
PREFIX = "cached."


def cache_paths(root, img_key):
    d = os.path.join(root, CACHE_DIR)
    return d, os.path.join(d, f"{img_key}.f16.npy"), os.path.join(d, f"{img_key}.json")


@torch.no_grad()
def build_cache(root, policy, device, batch=32, workers=4):
    """Compute embed_image for every frame of the dataset at `root`. Returns {img_key: path}."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
    ds = LeRobotDataset(f"local/{os.path.basename(os.path.normpath(root))}", root=root)
    cfg = policy.config
    vlm = policy.model.vlm_with_expert
    out = {}
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    for key in cfg.image_features:
        d, f_npy, f_json = cache_paths(root, key)
        os.makedirs(d, exist_ok=True)
        mm = None
        for b in loader:
            img = b[key].to(device)
            if cfg.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *cfg.resize_imgs_with_padding, pad_value=0)
            feat = vlm.embed_image(img * 2.0 - 1.0).float().cpu().numpy().astype(np.float16)
            if mm is None:
                mm = np.lib.format.open_memmap(f_npy + ".tmp", mode="w+", dtype=np.float16,
                                               shape=(len(ds), *feat.shape[1:]))
            mm[b["index"].numpy()] = feat
        mm.flush()
        del mm
        os.replace(f_npy + ".tmp", f_npy)
        with open(f_json, "w") as f:
            json.dump({"base": "lerobot/smolvla_base", "resize": cfg.resize_imgs_with_padding, "frames": len(ds),
                       "vision_dtype": str(vlm.get_vlm_model().vision_model.dtype)}, f)
        out[key] = f_npy
    return out


def has_cache(root, img_keys):
    return all(os.path.exists(cache_paths(root, k)[1]) for k in img_keys)


class CachedChunkDataset(torch.utils.data.Dataset):
    """Same samples as LeRobotDataset(delta_timestamps={"action": chunk}) but images replaced by cached features.
    Chunks past the episode end repeat the last action and are flagged in action_is_pad (as LeRobot does)."""

    def __init__(self, root, chunk_size, img_keys):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        ds = LeRobotDataset(f"local/{os.path.basename(os.path.normpath(root))}", root=root)
        cols = ds.hf_dataset.with_format("numpy")
        self.state = np.stack(cols["observation.state"]).astype(np.float32)
        self.action = np.stack(cols["action"]).astype(np.float32)
        self.sigma = np.asarray(cols["sigma_n"], np.float32).reshape(-1)
        self.ep = np.asarray(cols["episode_index"]).reshape(-1)
        self.index = np.asarray(cols["index"]).reshape(-1)
        task_idx = np.asarray(cols["task_index"]).reshape(-1)
        tasks = ds.meta.tasks  # DataFrame indexed by task string, column task_index
        names = {int(i): t for t, i in zip(tasks.index, tasks["task_index"])}
        self.task = [names[int(i)] for i in task_idx]
        # episode end (exclusive) for every frame
        ends = {}
        for i, e in enumerate(self.ep):
            ends[e] = i + 1
        self.end = np.array([ends[e] for e in self.ep])
        self.chunk = chunk_size
        self.feats = {k: np.load(cache_paths(root, k)[1], mmap_mode="r") for k in img_keys}

    def __len__(self):
        return len(self.state)

    def __getitem__(self, i):
        j = i + np.arange(self.chunk)
        pad = j >= self.end[i]
        j = np.minimum(j, self.end[i] - 1)
        item = {"observation.state": torch.from_numpy(self.state[i]), "action": torch.from_numpy(self.action[j]),
                "action_is_pad": torch.from_numpy(pad), "sigma_n": torch.tensor(self.sigma[i]),
                "task": self.task[i], "index": torch.tensor(self.index[i])}
        for k, f in self.feats.items():
            item[PREFIX + k] = torch.from_numpy(np.array(f[self.index[i]]))
        return item


def use_cached_features(policy):
    """Patch this policy instance so images come from batch['cached.<key>'] (already embed_image outputs)."""
    vlm = policy.model.vlm_with_expert
    keys = list(policy.config.image_features)

    def prepare_images(self, batch):
        imgs, masks = [], []
        for k in keys:
            f = batch[PREFIX + k]
            imgs.append(f)
            masks.append(torch.ones(f.shape[0], dtype=torch.bool, device=f.device))
        return imgs, masks

    def embed_image(self, feat):
        return feat.to(dtype=self.get_vlm_model().vision_model.dtype)  # what the real embed_image returns

    policy.prepare_images = types.MethodType(prepare_images, policy)
    vlm.embed_image = types.MethodType(embed_image, vlm)
    return policy
