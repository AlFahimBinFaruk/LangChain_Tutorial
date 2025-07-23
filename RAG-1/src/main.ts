import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { Document } from "@langchain/core/documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { Annotation, StateGraph } from "@langchain/langgraph";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import {
  ChatGoogleGenerativeAI,
  GoogleGenerativeAIEmbeddings,
} from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import "dotenv/config";

async function main() {
  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
  });

  const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004",
    taskType: TaskType.RETRIEVAL_DOCUMENT,
    title: "Test1",
  });

  const vectorStore = new Chroma(embeddings, {
    collectionName: "test-1",
  });

  const pTagSelector = "p";
  const cheerioLoader = new CheerioWebBaseLoader(
    "https://lilianweng.github.io/posts/2023-06-23-agent",
    { selector: pTagSelector }
  );
  const docs = await cheerioLoader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await splitter.splitDocuments(docs);

  // await vectorStore.addDocuments(allSplits);

  // see what ChatPromptTemplate does under the hood.
  // https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=f9443c0a-ce43-465f-b4fe-cc30b63cf915
  const promptTemplate = await pull<ChatPromptTemplate>("rlm/rag-prompt");

  const InputStateAnnotation = Annotation.Root({
    question: Annotation<string>,
  });

  const StateAnnotation = Annotation.Root({
    question: Annotation<string>,
    context: Annotation<[Document[]]>,
    answer: Annotation<string>,
  });

  const retrieve = async (state: typeof InputStateAnnotation.State) => {
    const embeddedQuery: any = await embeddings.embedQuery(state.question);
    // console.log("\n", embeddedQuery, "\n");
    const retrivedDocs = await vectorStore.similaritySearchVectorWithScore(
      [embeddedQuery],
      1
    );
    return { context: retrivedDocs };
  };

  const generate = async (state: typeof StateAnnotation.State) => {
    // state.context[0].map((doc) => console.log(doc.pageContent));
    const docsContent = state.context[0]
      .map((doc) => doc.pageContent)
      .join("\n");
    // console.log("context => ", state.context);
    const messages = await promptTemplate.invoke({
      question: state.question,
      context: docsContent,
    });
    const response = await llm.invoke(messages);
    return { answer: response.content };
  };

  const graph = new StateGraph(StateAnnotation)
    .addNode("retrieve", retrieve)
    .addNode("generate", generate)
    .addEdge("__start__", "retrieve")
    .addEdge("retrieve", "generate")
    .addEdge("generate", "__end__")
    .compile();

  let inputs = { question: "What is Task Decomposition?" };

  const result = await graph.invoke(inputs);
  console.log(result.answer);
}

main();
