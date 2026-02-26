from app.adapter.repository.compressor.bge import compress_vector
from app.adapter.repository.embeddings.bge import embed_text
from app.common.config import EMBEDDING_DIMENSION
from packages.shared.app_shared.schema.embedding import EmbeddingRequest, EmbeddingResponse


def generate_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    vectors = [compress_vector(embed_text(text, EMBEDDING_DIMENSION)) for text in request.texts]
    return EmbeddingResponse(vectors=vectors, dimension=EMBEDDING_DIMENSION if vectors else 0)
