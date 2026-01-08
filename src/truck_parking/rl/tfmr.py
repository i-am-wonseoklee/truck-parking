# src/truck_parking/rl/tfmr.py
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

_N_TTS = 10
_TT_CLS = 0
_TT_PARAM_LONCON = 1
_TT_PARAM_LATCON = 2
_TT_PARAM_GEOM = 3
_TT_PARAM_ART = 4
_TT_PARAM_ACT = 5
_TT_STATE = 6
_TT_OBST = 7
_TT_REF = 8
_TT_DIR = 9

_N_PARAM_TOKENS = 5


class Tfmr(BaseFeaturesExtractor):
    """Transformer-based feature extractor.

    CLS + [params] + [state] + [obsts] + [refs] + [dir] + pos-emb + type-emb
    -> Transformer -> CLS -> post-MLP -> features.
    """

    def __init__(self, ospace: spaces.Dict, cfg: Mapping[str, any]) -> None:
        super().__init__(ospace, features_dim=cfg["post_mlp"][-1])

        self._cfg = cfg
        sub = ospace.spaces

        self._n_obsts = sub["obsts"].shape[0]
        self._n_refs = sub["refs"].shape[0]

        dim_loncon = np.prod(sub["param_loncon"].shape)
        dim_latcon = np.prod(sub["param_latcon"].shape)
        dim_geom = np.prod(sub["param_geom"].shape)
        dim_art = np.prod(sub["param_art"].shape)
        dim_act = np.prod(sub["param_act"].shape)
        dim_state = np.prod(sub["state"].shape)
        dim_dir = np.prod(sub["dir"].shape)
        dim_obst = np.prod(sub["obsts"].shape[1:])
        dim_ref = np.prod(sub["refs"].shape[1:])

        n_param = _N_PARAM_TOKENS
        n_state = 1
        n_obst = self._n_obsts
        n_ref = self._n_refs
        n_dir = 1

        self._seq_len = n_param + n_state + n_obst + n_ref + n_dir
        self._seq_len_with_cls = self._seq_len + 1

        d = self._cfg["dim_mdl"]
        self._emb_loncon = nn.Linear(dim_loncon, d)
        self._emb_latcon = nn.Linear(dim_latcon, d)
        self._emb_geom = nn.Linear(dim_geom, d)
        self._emb_art = nn.Linear(dim_art, d)
        self._emb_act = nn.Linear(dim_act, d)
        self._emb_state = nn.Linear(dim_state, d)
        self._emb_dir = nn.Linear(dim_dir, d)
        self._emb_obst = nn.Linear(dim_obst, d)
        self._emb_ref = nn.Linear(dim_ref, d)
        self._cls = nn.Parameter(th.zeros(1, 1, d))
        nn.init.normal_(self._cls, mean=0.0, std=0.02)

        self._pos = nn.Parameter(th.zeros(1, self._seq_len_with_cls, d))
        nn.init.normal_(self._pos, mean=0.0, std=0.02)

        self._type_emb = nn.Embedding(_N_TTS, d)
        nn.init.normal_(self._type_emb.weight, mean=0.0, std=0.02)

        type_ids = self._build_type_ids()
        self.register_buffer("_type_ids", type_ids, persistent=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=self._cfg["num_heads"],
            dim_feedforward=self._cfg["dim_ff"],
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            layer_norm_eps=1e-5,
        )
        self._enc = nn.TransformerEncoder(enc_layer, num_layers=self._cfg["num_layers"])
        self._norm = nn.LayerNorm(d, eps=1e-5)
        self._post = self._mlp(d, self._cfg["post_mlp"])

        self._init_weights()

    def forward(self, obs: Dict[str, th.Tensor]) -> th.Tensor:
        """Build features from observation.

        Args:
            obs: An observation.

        Returns:
            Features (CLS token output after post-MLP).
        """
        tok, key_padding_mask = self._build_tokens(obs)
        tok = tok + self._pos[:, : tok.shape[1], :]
        tok = tok + self._type_emb(self._type_ids[:, : tok.shape[1]]).to(tok.dtype)
        feat = self._enc(tok, src_key_padding_mask=key_padding_mask)
        feat = self._norm(feat)
        return self._post(feat[:, 0, :])

    def _build_type_ids(self) -> th.Tensor:
        ids = [
            _TT_CLS,
            _TT_PARAM_LONCON,
            _TT_PARAM_LATCON,
            _TT_PARAM_GEOM,
            _TT_PARAM_ART,
            _TT_PARAM_ACT,
            _TT_STATE,
        ]
        ids += [_TT_OBST] * self._n_obsts
        ids += [_TT_REF] * self._n_refs
        ids.append(_TT_DIR)
        return th.tensor(ids, dtype=th.long).unsqueeze(0)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _build_tokens(self, obs: Dict[str, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
        B = obs["dir"].shape[0]
        device = obs["dir"].device

        # Params.
        loncon = self._emb_loncon(obs["param_loncon"].float().view(B, -1)).unsqueeze(1)
        latcon = self._emb_latcon(obs["param_latcon"].float().view(B, -1)).unsqueeze(1)
        geom = self._emb_geom(obs["param_geom"].float().view(B, -1)).unsqueeze(1)
        art = self._emb_art(obs["param_art"].float().view(B, -1)).unsqueeze(1)
        act = self._emb_act(obs["param_act"].float().view(B, -1)).unsqueeze(1)
        param_tokens = th.cat([loncon, latcon, geom, art, act], dim=1)

        # State.
        state_tok = self._emb_state(obs["state"].float().view(B, -1)).unsqueeze(1)

        # Obstacles (masked).
        obsts = obs["obsts"].float().view(B, self._n_obsts, -1)
        obst_tok = self._emb_obst(obsts)

        obst_valid = obs["obsts_mask"].to(th.bool)
        if obst_valid.ndim == 1:
            obst_valid = obst_valid.unsqueeze(0).expand(B, -1)
        obst_tok = obst_tok * obst_valid.unsqueeze(-1).float()

        # Refs.
        refs = obs["refs"].float().view(B, self._n_refs, -1)
        ref_tok = self._emb_ref(refs)

        # Dir (1).
        dir_tok = self._emb_dir(obs["dir"].float().view(B, -1)).unsqueeze(1)

        seq = th.cat([param_tokens, state_tok, obst_tok, ref_tok, dir_tok], dim=1)

        # True means PAD !!!.
        pad_params = th.zeros((B, _N_PARAM_TOKENS), dtype=th.bool, device=device)
        pad_state = th.zeros((B, 1), dtype=th.bool, device=device)
        pad_obsts = ~obst_valid.to(device)
        pad_refs = th.zeros((B, self._n_refs), dtype=th.bool, device=device)
        pad_dir = th.zeros((B, 1), dtype=th.bool, device=device)
        key_padding_mask = th.cat(
            [pad_params, pad_state, pad_obsts, pad_refs, pad_dir], dim=1
        )

        # CLS.
        cls = self._cls.expand(B, -1, -1)
        tok = th.cat([cls, seq], dim=1)

        pad_cls = th.zeros((B, 1), dtype=th.bool, device=device)
        key_padding_mask = th.cat([pad_cls, key_padding_mask], dim=1)

        return tok, key_padding_mask

    @staticmethod
    def _mlp(in_dim: int, sizes: Sequence[int]) -> nn.Sequential:
        layers: List[nn.Module] = []
        prev = in_dim
        for h in sizes:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.GELU())
            prev = int(h)
        return nn.Sequential(*layers)
