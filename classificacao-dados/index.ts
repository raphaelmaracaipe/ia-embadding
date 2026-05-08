import { QdrantVectorStore } from "@langchain/qdrant";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import * as dotenv from "dotenv";

dotenv.config();

export interface QdrantPointMetadata {
  id: number;
  source: string;
}

export interface SearchResult {
  content: string;
  metadata: QdrantPointMetadata;
  similarityScore: number;
}

export class QdrantSearchService {
  private vectorStore: QdrantVectorStore | null = null;

  constructor(
    private readonly qdrantUrl: string,
    private readonly collectionName: string,
  ) {}

  private async getVectorStore(): Promise<QdrantVectorStore> {
    if (this.vectorStore) {
      return this.vectorStore;
    }

    const embeddingDimensions = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      model: "gemini-embedding-001",
      taskType: TaskType.RETRIEVAL_DOCUMENT,
    });

    this.vectorStore = await QdrantVectorStore.fromExistingCollection(
      embeddingDimensions,
      {
        url: this.qdrantUrl,
        collectionName: this.collectionName,
        contentPayloadKey: "content",
        metadataPayloadKey: "metadata",
      },
    );

    return this.vectorStore;
  }

  public async searchSimilar(
    query: string,
    topK: number = 3,
  ): Promise<SearchResult[]> {
    if (!query || query.trim() === "") {
      throw new Error("A query de busca não pode estar vazia.");
    }

    try {
      const vectorStore = await this.getVectorStore();
      const results = await vectorStore.similaritySearch(query, topK);

      return results.map((document, score) => ({
        content: document.pageContent,
        metadata: document.metadata as QdrantPointMetadata,
        similarityScore: score,
      }));
    } catch (error) {
      console.error(
        "[QdrantSearchService] Falha ao realizar busca vetorial:",
        error,
      );
      throw new Error(
        "Não foi possível recuperar os documentos da base de conhecimento.",
      );
    }
  }
}

async function runSearch() {
  const qdrantUrl = "http://localhost:6333";
  const collectionName = "exemplo-collection";
  const searchService = new QdrantSearchService(qdrantUrl, collectionName);

  try {
    const query = "JavaScript";
    const result = await searchService.searchSimilar(query, 2);

    console.log(`✅ Busca concluída para: "${query}"\n`);

    result.forEach((result, index) => {
      console.log(
        `--- Resultado ${index + 1} (Score: ${result.similarityScore.toFixed(4)}) ---`,
      );
      console.log(`Conteúdo: ${result.content}`);
      console.log(
        `Fonte: ${result.metadata.source} (ID: ${result.metadata.id})\n`,
      );
    });
  } catch (error) {
    console.error("An error occurred:", error);
  }
}

runSearch();
