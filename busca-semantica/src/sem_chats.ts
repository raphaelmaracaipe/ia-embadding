import { Document } from "@langchain/core/documents";
import { ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { TaskType } from "@google/generative-ai";
import { randomUUID } from "crypto";
import process from "process";

console.log("aaa => ", process.env.GOOGLE_API_KEY);

const qrdrantUrl = "http://localhost:6333";
const collectionName = "exemplo-collection";

const embeddingDimensions = new GoogleGenerativeAIEmbeddings({
  apiKey: "....",
  model: "gemini-embedding-001",
  taskType: TaskType.RETRIEVAL_DOCUMENT,
});

async function indexDocuments() {
  console.log("📚 Indexando documentos...\n");

  const documents = [
    "Machine Learning é um subcampo da Inteligência Artificial que permite que os sistemas aprendam e melhorem a partir da experiência.",
    "Python é uma linguagem de programação versátil usada amplamente em Data Science e Machine Learning.",
    "Redes neurais são modelos computacionais inspirados no cérebro que aprendem padrões nos dados.",
    "A análise de dados envolve extrair insights significativos de grandes volumes de informações.",
    "JavaScript é uma linguagem de programação utilizada principalmente para desenvolvimento web.",
    "Deep Learning é uma subárea do Machine Learning que usa redes neurais profundas.",
    "Processamento de linguagem natural permite que computadores entendam texto humano.",
    "Dados estruturados são organizados em tabelas e bancos de dados relacionais.",
  ];

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 500,
    chunkOverlap: 50,
  });

  const docs: Document[] = [];
  for (let i = 0; i < documents.length; i++) {
    docs.push(
      new Document({
        id: randomUUID(),
        pageContent: documents[i],
        metadata: { id: i, source: `exemplo` },
      }),
    );
  }

  const vectorStore = await QdrantVectorStore.fromDocuments(
    docs,
    embeddingDimensions,
    {
      url: qrdrantUrl,
      collectionName: collectionName,
    },
  );

  return vectorStore;
}

async function semanticSearch(query: string, topK: number = 3) {
  console.log(" 🔍 Realizando busca semântica...\n");

  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embeddingDimensions,
    {
      url: qrdrantUrl,
      collectionName: collectionName,
    },
  );

  const results = await vectorStore.similaritySearch(query, topK);

  console.log(`📍 Top ${topK} resultados mais relevantes:\n`);
  results.forEach((result, index) => {
    console.log(`${index + 1}. ${result.pageContent}`);
    console.log(`   Metadados: ${JSON.stringify(result.metadata)}\n`);
  });

  return results;
}

async function semanticSearchWithScores(query: string, topK: number = 3) {
  console.log(" 🔍 Realizando busca semântica com scores...\n");

  const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddingDimensions, {
    url: qrdrantUrl,
    collectionName: collectionName,
  });

  const results = await vectorStore.similaritySearchWithScore(query, topK);

  console.log(`📊 Resultados com scores de similaridade:\n`);
  results.forEach(([doc, score], index) => {
    console.log(`${index + 1}. Similaridade: ${(score * 100).toFixed(2)}%`);
    console.log(`   ${doc.pageContent}\n`);
  });

  return results;
}

async function main() {
  try {
    console.log("🚀 Iniciando exemplo de busca semântica com Qdrant\n");

    // await indexDocuments();

    // await semanticSearch("python", 3);

    // await semanticSearchWithScores("onde usar python?", 3);
  } catch (err) {
    console.error(err);
  }
}

main();
