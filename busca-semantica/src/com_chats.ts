import { Document } from "@langchain/core/documents";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { TaskType } from "@google/generative-ai";
import { randomUUID } from "crypto";
import process from "process";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";

const qrdrantUrl = "http://localhost:6333";
const collectionName = "exemplo-collection";

const llm = new ChatGoogleGenerativeAI({
  apiKey: "....",
  model: "gemini-2.0-flash",
  temperature: 0.7,
  maxOutputTokens: 1024,
});
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

async function generateResponseWithRag(userMessage: string) {
  console.log("\n🔍 Buscando documentos relevantes...");

  const { context, scores } = await retrieveContext(userMessage, 3);
  console.log(`📄 Documentos encontrados: ${context.length}`);
  console.log(
    `📊 Relevância dos documentos: ${scores.map((s) => (s * 100).toFixed(0) + "%").join(", ")}\n`,
  );

  const promptTemplate = PromptTemplate.fromTemplate(`
  Você é um assistente especializado em tecnologia e desenvolvimento.
  
  Use o seguinte contexto para responder a pergunta do usuário. Se o contexto não contiver informação suficiente, diga claramente que você não tem essa informação na base de conhecimento.
  
  Contexto:
  {context}
  
  Pergunta do usuário: {question}
  
  Responda de forma clara, concisa e útil. Se possível, cite qual parte do contexto você usou. Caso nao tenha informação suficiente, responda que não tem essa informação nao invente.
  `);

  const chain = RunnableSequence.from([
    promptTemplate,
    llm
  ]);

  console.log("🤖 Gerando resposta com RAG... userMessage: " + userMessage);
  const response = await chain.invoke({
    context: context,
    question: userMessage,
  })
  console.log("✅ Resposta gerada com sucesso!\n");

  return {
    response: response.content,
    relevanceScores: scores,
    sourcesUsed: scores.length
  }
}

async function retrieveContext(query: string, topK: number = 3) {
  console.log(`🔎 Realizando busca vetorial para: "${query}"\n`);
  const vectorStore = new QdrantVectorStore(embeddingDimensions, {
    url: qrdrantUrl,
    collectionName: collectionName,
  });
  console.log("✅ Conectado ao Qdrant Vector Store\n");

  const relevantDocs = await vectorStore.similaritySearchWithScore(query, topK);
  console.log(`📊 Documentos relevantes encontrados: ${relevantDocs.length}\n`);

  const context = relevantDocs
    .map(
      ([doc, score]) =>
        `[Relevância: ${(score * 100).toFixed(0)}%]\n${doc.pageContent}`,
    )
    .join("\n\n---\n\n");
  console.log("📚 Contexto construído para o LLM ");

  return {
    context,
    documents: relevantDocs.map(([doc]) => doc),
    scores: relevantDocs.map(([_, score]) => score),
  };
}

async function startInteractiveChat() {
  console.log("🚀 Exemplos de Chat com Gemini\n");
  console.log("=".repeat(70) + "\n");

  // await indexDocuments();

  // Exemple 01
  const question1 = "Qual é a melhor linguagem de programação para data science?";
  console.log(`📌 Pergunta 1: "${question1}"\n`);
  const answer1 = await generateResponseWithRag(question1);

  console.log("🤖 Resposta:\n" + answer1.response);
  console.log(
    `\n📊 Relevância média: ${(
      (answer1.relevanceScores.reduce((a, b) => a + b, 0) /
        answer1.relevanceScores.length) *
      100
    ).toFixed(0)}%\n`,
  );

  console.log("=".repeat(70) + "\n");
}

async function main() {
  try {
    await startInteractiveChat();
  } catch (error) {
    console.error("❌ Erro fatal:", error);
    process.exit(1);
  }
}

main();
