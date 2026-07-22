from dwutils import mirror

MODEL_SHORT_NAMES = {
    "all-MiniLM-L6-v2": "minilmL6",
    "all-mpnet-base-v2": "mpnetB",
    "bge-base-en-v1.5": "bgeB",
    "bge-m3": "bgeM3",
    "bge-reranker-base": "bgeR",
    "ms-marco-MiniLM-L-6-v2": "marcoL6",
    "DeBERTa-v3-large-mnli-fever-anli-ling-wanli": "DeBERTa-mnli",
    "eu.anthropic.claude-sonnet-4-6": "sonnet",
    "eu.anthropic.claude-haiku-4-5-20251001-v1:0": "haiku",
}

mirror.clone_huggingface_model(
    model="huggingface.co/sentence-transformers/all-mpnet-base-v2",
    dir="/home/dw-user-efs/onesnapshot-nlp/models",
)

mirror.clone_huggingface_model(
    model="huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    dir="/home/dw-user-efs/onesnapshot-nlp/models",
)

mirror.clone_huggingface_model(
    model="huggingface.co/BAAI/bge-m3",
    dir="/home/dw-user-efs/onesnapshot-nlp/models",
)

mirror.clone_huggingface_model(
    model="huggingface.co/BAAI/bge-base-en-v1.5",
    dir="/home/dw-user-efs/onesnapshot-nlp/models",
)

mirror.clone_huggingface_model(
    model="huggingface.co/BAAI/bge-reranker-base",
    dir="/home/dw-user-efs/onesnapshot-nlp/models",
)

mirror.clone_huggingface_model(
    model="huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2",
    dir="/home/dw-user-efs/onesnapshot-nlp/models",
)

mirror.clone_huggingface_model(
    model="huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
    dir="/home/dw-user-efs/onesnapshot-nlp/models",
)

mirror.clone_huggingface_model(
    model="huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",
    dir="/home/dw-user-efs/onesnapshot-nlp/models",
)
