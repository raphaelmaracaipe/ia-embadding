import { RAG } from "./rag.js";

export class Chat {
  startInteractionChat() {
    console.log("🚀 Exemplos de Chat com Gemini\n");
    console.log("=".repeat(70) + "\n");

    const question1 =
      "Qual é a melhor linguagem de programação para data science?";
    console.log(`📌 Pergunta 1: "${question1}"\n`);

    const answer1 = RAG.generateResponseWithRag(question1);
  }
}
