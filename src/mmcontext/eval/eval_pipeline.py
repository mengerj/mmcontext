# embedding_benchmark/eval_pipeline.py
import importlib
import pkgutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

import mmcontext.eval as ev_pkg
from mmcontext.eval.registry import get as get_evaluator

# from mmcontext.adata_utils import collect_adata_subset
from mmcontext.eval.utils import LabelKind, LabelSpec
from mmcontext.file_utils import save_table

# ---- discover all evaluator modules so decorators run ---------------
for m in pkgutil.walk_packages(ev_pkg.__path__, prefix=ev_pkg.__name__ + "."):
    importlib.import_module(m.name)


def run_eval_suite(cfg) -> None:
    """
    Run the evaluation suite.

    Is called from the eval.py script.
    """
    for ds_cfg in cfg.datasets:
        label_specs = [LabelSpec(n, LabelKind.BIO) for n in ds_cfg.bio_label_list] + [
            LabelSpec(n, LabelKind.BATCH) for n in ds_cfg.batch_label_list
        ]
        for model_cfg in cfg.models:
            rows = []
            scib_rows = []  # Separate storage for scIB results
            model_id = model_cfg.source
            text_only = model_cfg.get("text_only", False)
            if text_only:
                model_id = model_id + "_text_only"
            emb_dir = Path(cfg.output.root) / ds_cfg.name / Path(model_id).name.replace("/", "_")

            # ── load shared artefacts *once* per model ───────────────────────
            emb_df = pd.read_parquet(emb_dir / "embeddings.parquet")
            E1 = np.vstack(emb_df["embedding"].to_numpy())
            adata = ad.read_zarr(emb_dir / "subset.zarr")

            # ──── Run ScibBundle separately if in suite ──────────────────────
            if "scib" in cfg.eval.suite:
                print(f"Running ScibBundle for {ds_cfg.name}/{model_id}")
                try:
                    ScibClass = get_evaluator("scib")
                    scib_evaluator = ScibClass()

                    scib_results = scib_evaluator.compute_dataset_model(
                        emb1=E1,
                        adata=adata,
                        dataset_name=ds_cfg.name,
                        model_id=model_id,
                        bio_labels=ds_cfg.bio_label_list,
                        batch_labels=ds_cfg.batch_label_list,
                        **cfg.eval,
                    )

                    scib_rows.extend(scib_results)
                    print(f"✓ ScibBundle completed for {ds_cfg.name}/{model_id}")

                except Exception as e:
                    print(f"✗ ScibBundle failed for {ds_cfg.name}/{model_id}: {e}")
                    # Add error entry
                    scib_rows.append(
                        {
                            "dataset": ds_cfg.name,
                            "model": model_id,
                            "bio_label": "unknown",
                            "batch_label": "unknown",
                            "metric": "scib/error",
                            "value": str(e),
                            "data_id": "",
                            "hvg": "",
                            "type": "",
                        }
                    )

            # ──── Run regular evaluators (excluding scib) ────────────────────
            # Filter out scib from regular evaluators
            regular_evaluators = [ev_name for ev_name in cfg.eval.suite if ev_name != "scib"]

            # try to load label embeddings only once
            label_emb_cache = {}  # (kind, name) → ndarray | None
            # ------------- ITERATE over labels AND evaluators --------------
            for label_spec in label_specs:
                if label_spec.name not in adata.obs.columns:
                    continue  # skip silently
                y = adata.obs[label_spec.name].to_numpy()
                if (label_spec.kind, label_spec.name) not in label_emb_cache:
                    prefix = "bio_label_embeddings" if label_spec.kind == LabelKind.BIO else "batch_label_embeddings"
                    path = emb_dir / f"{prefix}_{label_spec.name}.parquet"
                    if path.exists():
                        df2 = pd.read_parquet(path)
                        label_emb_cache[(label_spec.kind, label_spec.name)] = np.vstack(df2["embedding"].to_numpy())
                    else:
                        label_emb_cache[(label_spec.kind, label_spec.name)] = None
                E2 = label_emb_cache[(label_spec.kind, label_spec.name)]

                for ev_name in regular_evaluators:
                    EvClass = get_evaluator(ev_name)
                    ev = EvClass()

                    if ev.requires_pair and E2 is None:
                        continue

                    result = ev.compute(
                        E1,
                        emb2=E2,
                        labels=y,
                        adata=adata,
                        label_kind=label_spec.kind,  # evaluators may ignore
                        label_key=label_spec.name,  # evaluators may ignore
                        **cfg.eval,
                    )

                    for key, val in result.items():
                        rows.append(
                            {
                                "dataset": ds_cfg.name,
                                "model": model_id,
                                "label": label_spec.name,
                                "label_kind": label_spec.kind,
                                "metric": f"{ev_name}/{key}",
                                "value": val,
                            }
                        )

                    if ev.produces_plot:
                        # …/eval/<Evaluator>/<label-name>/figure.png
                        plot_dir = (
                            emb_dir / "eval" / ev_name / label_spec.name  # <— NEW: sub-folder per label value
                        )
                        plot_dir.mkdir(parents=True, exist_ok=True)

                        ev.plot(
                            E1,
                            out_dir=plot_dir,
                            emb2=E2,
                            labels=y,  # ground-truth for *this* label column
                            adata=adata,
                            label_kind=label_spec.kind,  # "bio"  or  "batch"
                            label_key=label_spec.name,  # column name (e.g. "celltype")
                            **cfg.eval,  # forward any extra Hydra knobs
                        )

            # ──── Save results ────────────────────────────────────────────────
            # Save regular evaluator results
            if rows:
                out_df = pd.DataFrame(rows)
                save_table(out_df, emb_dir / "eval/metrics", fmt="csv")

            # Save ScibBundle results separately
            if scib_rows:
                scib_df = pd.DataFrame(scib_rows)
                save_table(scib_df, emb_dir / "eval/scib_metrics", fmt="csv")
                print(f"✓ ScibBundle results saved to {emb_dir / 'eval/scib_metrics.csv'}")
