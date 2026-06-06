"""Search spaces — the editable ``train.py`` constants per track and their candidate values.

Keys MUST match the UPPER_CASE constant names at the top of each track's ``train.py``.
``higher_better`` tells the optimizer how to rank ``primary_metric`` (all three tracks emit
``primary_metric`` so that higher is better — generative emits -eval_loss).
"""

SPACES = {
    "classification": {
        "higher_better": True,
        "extra_greps": ["test_macro_f1", "kappa", "peak_vram_mb"],
        "space": {
            "LEARNING_RATE": [1e-5, 2e-5, 3e-5, 5e-5],
            "BATCH_SIZE": [8, 16, 32],
            "DROPOUT": [0.1, 0.2, 0.3],
            "WEIGHT_DECAY": [0.01, 0.1],
            "WARMUP_RATIO": [0.06, 0.1],
            "LABEL_SMOOTHING": [0.0, 0.05, 0.1],
            "USE_PROVENANCE_WEIGHTS": [False, True],
            "BASE_MODEL": [
                "emilyalsentzer/Bio_ClinicalBERT",
                "dmis-lab/biobert-base-cased-v1.1",
                "mental/mental-bert-base-uncased",
            ],
        },
    },
    "nsp": {
        "higher_better": True,
        "extra_greps": ["test_roc_auc", "mrr", "recall@5", "peak_vram_mb"],
        "space": {
            "LEARNING_RATE": [1e-5, 2e-5, 3e-5],
            "BATCH_SIZE": [8, 16, 32],
            "POS_WEIGHT": [1.0, 2.0, 4.0],
            "WEIGHT_DECAY": [0.01, 0.1],
            "WARMUP_RATIO": [0.06, 0.1],
            "BASE_MODEL": [
                "dmis-lab/biobert-base-cased-v1.1",
                "emilyalsentzer/Bio_ClinicalBERT",
                "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
            ],
        },
    },
    "generative": {
        "higher_better": True,
        "extra_greps": ["eval_ppl", "test_eval_loss", "trainable_params_M", "peak_vram_mb"],
        "space": {
            "LORA_R": [8, 16, 32, 64],
            "LORA_ALPHA": [16, 32, 64],
            "LORA_DROPOUT": [0.0, 0.05, 0.1],
            "LEARNING_RATE": [1e-4, 2e-4, 3e-4],
            "GRAD_ACCUM": [8, 16, 32],
            "MICRO_BATCH": [1, 2],
            "WARMUP_RATIO": [0.03, 0.06],
        },
    },
}
