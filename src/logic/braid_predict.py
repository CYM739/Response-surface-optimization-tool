# src/logic/braid_predict.py
"""
Prediction engine for the v19 BRAID surface model.

Loads a trained V19BRAIDPredictor checkpoint + all feature files,
exposes predict_surface(drug1, drug2, cell_line, mono_params) for
interactive use in the Streamlit app.

All heavy imports (torch, torch_geometric, rdkit) are deferred
to load time so the module can be imported without side effects.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths (override via env) ────────────────────────────────────────────────
_SYNERGY_ROOT = Path(os.environ.get(
    "SYNERGY_ROOT", "C:/Max/Github/synergy_docker"))
_DATA_PATH = _SYNERGY_ROOT / "data"
_BRAID_PATH = _SYNERGY_ROOT / "braid_surface_prediction" / "data"
_CKPT_DIR = _SYNERGY_ROOT / "braid_surface_prediction"

# ── Normalisation constants (computed from training_index.parquet) ───────────
# These are frozen from the training run and must match the checkpoint.
PARAM_MEAN = np.array([100.903015, 18.304684, 5.643743], dtype=np.float32)
PARAM_STD = np.array([3.335608, 29.621043, 9.447192], dtype=np.float32)
MONO_MEAN = np.array([0.897883, 4.039117, 0.554314, 4.017952], dtype=np.float32)
MONO_STD = np.array([3.958147, 2.991041, 4.132876, 2.929012], dtype=np.float32)


def _ensure_synergy_on_path():
    """Add synergy_docker to sys.path so we can import model code."""
    root = str(_SYNERGY_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class BRAIDPredictionEngine:
    """
    Loads v19 model + feature files once, then provides fast inference.
    Designed to be cached in st.session_state.
    """

    def __init__(self, checkpoint: str = "best_v19_cold_drug.pth"):
        _ensure_synergy_on_path()
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._checkpoint_name = checkpoint

        # Load all feature data
        self._load_features()

        # Build and load model
        self._build_model()

        # Load the QC database for mono-param lookup
        self._load_qc_index()

    # ── Feature loading ──────────────────────────────────────────────────────

    def _load_features(self):
        import torch
        from run_v18_braid import (
            load_molformer, load_ppi, load_cell_features,
            load_pathway_mask, load_mutations, load_drug_targets,
            load_drug_graphs,
        )
        from run_v18_braid_augmented import load_chemberta, load_tanimoto

        # Temporarily override DATA_PATH for the loaders
        import run_v18_braid
        orig_data = run_v18_braid.DATA_PATH
        run_v18_braid.DATA_PATH = str(_DATA_PATH)
        run_v18_braid.CONFIG["DATA_PATH"] = str(_DATA_PATH)

        # Monkey-patch the p() helper
        orig_p = run_v18_braid.p
        run_v18_braid.p = lambda fname: os.path.join(str(_DATA_PATH), fname)

        self.molformer = load_molformer()
        self.chemberta = load_chemberta()
        self.tanimoto = load_tanimoto()
        self.ppi = load_ppi()
        self.cell_expr, self.svd_df, self._expr_full, self._gene_names = load_cell_features()
        self.pathway_mask = load_pathway_mask(self.cell_expr.columns.tolist())
        self.mut_df = load_mutations()
        self.target_df = load_drug_targets(self.molformer.index)
        self.drug_graphs = load_drug_graphs(self.molformer.index)

        # Restore
        run_v18_braid.DATA_PATH = orig_data
        run_v18_braid.p = orig_p

        # Set normalisation on the dataset class
        from run_v19_braid import V19Dataset
        V19Dataset.PARAM_MEAN = torch.tensor(PARAM_MEAN)
        V19Dataset.PARAM_STD = torch.tensor(PARAM_STD)
        V19Dataset.MONO_MEAN = torch.tensor(MONO_MEAN)
        V19Dataset.MONO_STD = torch.tensor(MONO_STD)

    def _build_model(self):
        import torch
        from run_v19_braid import V19BRAIDPredictor, CONFIG as V19_CONFIG

        # Match training config
        V19_CONFIG["TARGET_DIM"] = self.target_df.shape[1]
        V19_CONFIG["PPI_DIM"] = self.ppi.shape[1]

        self.model = V19BRAIDPredictor(V19_CONFIG, self.pathway_mask).to(self.device)

        ckpt_path = _CKPT_DIR / self._checkpoint_name
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def _load_qc_index(self):
        """Load braid_labels_qc.parquet for mono-param auto-fill."""
        qc_path = _BRAID_PATH / "braid_labels_qc.parquet"
        if qc_path.exists():
            self.qc_df = pd.read_parquet(qc_path)
            self.qc_df["_d1"] = self.qc_df["drug1"].str.upper().str.strip()
            self.qc_df["_d2"] = self.qc_df["drug2"].str.upper().str.strip()
            self.qc_df["_cl"] = self.qc_df["cell_line"].str.upper().str.strip()
        else:
            self.qc_df = None

    # ── Public API ───────────────────────────────────────────────────────────

    def get_available_drugs(self):
        """Drugs with MolFormer embeddings (required for prediction)."""
        return sorted(self.molformer.index.tolist())

    def get_available_cells(self):
        """Cell lines with expression data (required for prediction)."""
        return sorted(self.cell_expr.index.tolist())

    def get_cell_name(self, depmap_id):
        """Resolve DepMap_ID → cell line name via QC database."""
        if self.qc_df is not None:
            match = self.qc_df[self.qc_df["DepMap_ID"] == depmap_id]
            if not match.empty:
                return match.iloc[0]["cell_line"]
        return depmap_id

    def lookup_mono_params(self, drug1, drug2, cell_line):
        """
        Look up monotherapy params from the QC database.
        Returns dict {EC50_1, h1, EC50_2, h2} or None if not found.
        cell_line can be either a name or DepMap_ID.
        """
        if self.qc_df is None:
            return None

        d1u = drug1.upper().strip()
        d2u = drug2.upper().strip()
        clu = cell_line.upper().strip()

        # Try both drug orders, match on cell_line or DepMap_ID
        for a, b in [(d1u, d2u), (d2u, d1u)]:
            mask = (
                (self.qc_df["_d1"] == a) &
                (self.qc_df["_d2"] == b) &
                ((self.qc_df["_cl"] == clu) | (self.qc_df["DepMap_ID"] == cell_line))
            )
            if mask.any():
                row = self.qc_df[mask].iloc[0]
                # If drug order was swapped, swap the mono params
                if a == d2u:
                    return {
                        "EC50_1": float(row["EC50_2"]),
                        "h1": float(row["h2"]),
                        "EC50_2": float(row["EC50_1"]),
                        "h2": float(row["h1"]),
                    }
                return {
                    "EC50_1": float(row["EC50_1"]),
                    "h1": float(row["h1"]),
                    "EC50_2": float(row["EC50_2"]),
                    "h2": float(row["h2"]),
                }
        return None

    def predict_surface(self, drug1, drug2, cell_line, mono_params, n_points=60):
        """
        Predict the BRAID surface for a drug combination.

        Args:
            drug1:       drug name (must be in molformer index)
            drug2:       drug name (must be in molformer index)
            cell_line:   DepMap_ID (must be in cell_expr index)
            mono_params: dict with {EC50_1, h1, EC50_2, h2}
            n_points:    grid resolution for surface reconstruction

        Returns:
            dict with {E0, Einf, kappa, EC50_1, h1, EC50_2, h2,
                       d1_arr, d2_arr, V_grid, kappa_label, kappa_color}
        """
        import torch
        from torch_geometric.data import Data, Batch as PyGBatch

        # ── Build single-sample batch ────────────────────────────────────────
        d1_vec = (torch.tensor(self.molformer.loc[drug1].values.copy(), dtype=torch.float)
                  if drug1 in self.molformer.index else torch.zeros(768))
        d2_vec = (torch.tensor(self.molformer.loc[drug2].values.copy(), dtype=torch.float)
                  if drug2 in self.molformer.index else torch.zeros(768))

        d1_cb = (torch.tensor(self.chemberta.loc[drug1].values.copy(), dtype=torch.float)
                 if drug1 in self.chemberta.index else torch.zeros(384))
        d2_cb = (torch.tensor(self.chemberta.loc[drug2].values.copy(), dtype=torch.float)
                 if drug2 in self.chemberta.index else torch.zeros(384))

        if drug1 in self.tanimoto.index and drug2 in self.tanimoto.columns:
            tan_val = float(self.tanimoto.loc[drug1, drug2])
        else:
            tan_val = 0.0

        d1_g = self.drug_graphs.get(drug1)
        d2_g = self.drug_graphs.get(drug2)
        empty = Data(x=torch.zeros(1, 30), edge_index=torch.zeros(2, 0, dtype=torch.long))
        d1_g = d1_g if d1_g is not None else empty
        d2_g = d2_g if d2_g is not None else empty

        d1_t = (torch.tensor(self.target_df.loc[drug1].values.copy(), dtype=torch.float)
                if drug1 in self.target_df.index else torch.zeros(self.target_df.shape[1]))
        d2_t = (torch.tensor(self.target_df.loc[drug2].values.copy(), dtype=torch.float)
                if drug2 in self.target_df.index else torch.zeros(self.target_df.shape[1]))

        cid = cell_line
        c_expr = (torch.tensor(self.cell_expr.loc[cid].values.copy(), dtype=torch.float)
                  if cid in self.cell_expr.index else torch.zeros(self.cell_expr.shape[1]))
        c_svd = (torch.tensor(self.svd_df.loc[cid].values.copy(), dtype=torch.float)
                 if cid in self.svd_df.index else torch.zeros(self.svd_df.shape[1]))
        c_ppi = (torch.tensor(self.ppi.loc[cid].values.copy(), dtype=torch.float)
                 if cid in self.ppi.index else torch.zeros(self.ppi.shape[1]))
        c_mut = (torch.tensor(self.mut_df.loc[cid].values.copy(), dtype=torch.float)
                 if (len(self.mut_df) > 0 and cid in self.mut_df.index)
                 else torch.zeros(200))

        # Mono params → normalised input token
        lec1 = float(np.log(mono_params["EC50_1"]))
        h1 = float(mono_params["h1"])
        lec2 = float(np.log(mono_params["EC50_2"]))
        h2 = float(mono_params["h2"])
        mono_raw = torch.tensor([lec1, h1, lec2, h2], dtype=torch.float)
        mono_feat = (mono_raw - torch.tensor(MONO_MEAN)) / torch.tensor(MONO_STD)

        # Assemble batch (B=1)
        batch = {
            "d1_vec":    d1_vec.unsqueeze(0).to(self.device),
            "d2_vec":    d2_vec.unsqueeze(0).to(self.device),
            "d1_cb":     d1_cb.unsqueeze(0).to(self.device),
            "d2_cb":     d2_cb.unsqueeze(0).to(self.device),
            "tanimoto":  torch.tensor([tan_val]).to(self.device),
            "d1_g":      PyGBatch.from_data_list([d1_g]).to(self.device),
            "d2_g":      PyGBatch.from_data_list([d2_g]).to(self.device),
            "d1_target": d1_t.unsqueeze(0).to(self.device),
            "d2_target": d2_t.unsqueeze(0).to(self.device),
            "cell_expr": c_expr.unsqueeze(0).to(self.device),
            "cell_svd":  c_svd.unsqueeze(0).to(self.device),
            "cell_ppi":  c_ppi.unsqueeze(0).to(self.device),
            "cell_mut":  c_mut.unsqueeze(0).to(self.device),
            "mono_feat": mono_feat.unsqueeze(0).to(self.device),
        }

        # ── Inference ────────────────────────────────────────────────────────
        with torch.no_grad():
            pred_norm = self.model(batch)  # [1, 3] normalised [E0, Einf, kappa]

        # Denormalise
        pm = torch.tensor(PARAM_MEAN).to(self.device)
        ps = torch.tensor(PARAM_STD).to(self.device)
        pred_dn = (pred_norm * ps + pm).cpu().numpy().flatten()

        E0_pred = float(pred_dn[0])
        Einf_pred = float(pred_dn[1])
        kappa_pred = float(pred_dn[2])

        # ── Reconstruct surface ──────────────────────────────────────────────
        EC50_1 = mono_params["EC50_1"]
        EC50_2 = mono_params["EC50_2"]
        h1_val = mono_params["h1"]
        h2_val = mono_params["h2"]

        d1_arr = np.linspace(0, 4.0 * EC50_1, n_points)
        d2_arr = np.linspace(0, 4.0 * EC50_2, n_points)
        D1, D2 = np.meshgrid(d1_arr, d2_arr)

        with np.errstate(over="ignore", invalid="ignore"):
            T1 = np.clip((D1 / EC50_1) ** h1_val, 0, 1e6)
            T2 = np.clip((D2 / EC50_2) ** h2_val, 0, 1e6)
            denom = np.clip(1.0 + T1 + T2 + kappa_pred * T1 * T2, 1e-9, None)
            V_grid = Einf_pred + (E0_pred - Einf_pred) / denom

        # Kappa classification
        if kappa_pred > 1:
            k_label, k_color = "Synergistic", "#27ae60"
        elif kappa_pred < -1:
            k_label, k_color = "Antagonistic", "#e74c3c"
        else:
            k_label, k_color = "Additive", "#2980b9"

        return {
            "E0": E0_pred,
            "Einf": Einf_pred,
            "kappa": kappa_pred,
            "EC50_1": EC50_1,
            "h1": h1_val,
            "EC50_2": EC50_2,
            "h2": h2_val,
            "d1_arr": d1_arr,
            "d2_arr": d2_arr,
            "V_grid": V_grid,
            "kappa_label": k_label,
            "kappa_color": k_color,
            "drug1": drug1,
            "drug2": drug2,
            "cell_line": cell_line,
        }
