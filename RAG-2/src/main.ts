import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { TaskType } from "@google/generative-ai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import "cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { z } from "zod";
import { tool } from "@langchain/core/tools";

import {
  AIMessage,
  HumanMessage,
  SystemMessage,
  ToolMessage,
} from "@langchain/core/messages";
import { MessagesAnnotation } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";

import { StateGraph } from "@langchain/langgraph";
import { toolsCondition } from "@langchain/langgraph/prebuilt";

import { BaseMessage, isAIMessage } from "@langchain/core/messages";

import { MemorySaver } from "@langchain/langgraph";

import { createReactAgent } from "@langchain/langgraph/prebuilt";

import "dotenv/config";




const prettyPrint = (message: BaseMessage) => {
  let txt = `[${message._getType()}]: ${message.content}`;
  if ((isAIMessage(message) && message.tool_calls?.length) || 0 > 0) {
    const tool_calls = (message as AIMessage)?.tool_calls
      ?.map((tc) => `- ${tc.name}(${JSON.stringify(tc.args)})`)
      .join("\n");
    txt += ` \nTools: \n${tool_calls}`;
  }
  console.log(txt);
};






async function main() {




  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
  });

  const embeddings = new GoogleGenerativeAIEmbeddings({
    model: "text-embedding-004", // 768 dimensions
  });

  const pinecone = new PineconeClient();
  const pineconeIndex = pinecone.Index("rag-2");

  const vectorStore = new PineconeStore(embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });

  // Scrapping the page.
  const pTagSelector = "p";
  const cheerioLoader = new CheerioWebBaseLoader(
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    {
      selector: pTagSelector,
    }
  );

  const docs = await cheerioLoader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await splitter.splitDocuments(docs);
  // Adding the retrived info to DB.
  // await vectorStore.addDocuments(allSplits);

  // What i need to give input to get result(input args my retrive-tool expects).
  const retrieveSchema = z.object({ query: z.string() });

  const retrieve = tool(
    async ({ query }) => {
      const retrievedDocs = await vectorStore.similaritySearch(query, 2);
      const serialized = retrievedDocs
        .map(
          (doc) => `Source: ${doc.metadata.source}\nContent: ${doc.pageContent}`
        )
        .join("\n");
      return [serialized, retrievedDocs];
    },
    {
      name: "retrieve",
      description: "Retrieve information related to a query.",
      schema: retrieveSchema,
      responseFormat: "content_and_artifact",
    }
  );

  // Step 1: Generate an AIMessage that may include a tool-call to be sent.
  /**
   * This basically tells that we need to invoke the tool or not.
   */
  async function queryOrRespond(state: typeof MessagesAnnotation.State) {
    const llmWithTools = llm.bindTools([retrieve]);
    /**
     * the structure of the "response" will be something like
     * { 
          tool_calls: [
            { name: "retrieve", args: { query: "Your question" }, ... }
          ]
        }
      depending on this response we will call the tool(toolConditions handles it.)
     */
    const response = await llmWithTools.invoke(state.messages);
    // MessagesState appends messages to state instead of overwriting
    return { messages: [response] };
  }

  // Step 2: Execute the retrieval.
  const tools = new ToolNode([retrieve]);

  // Step 3: Generate a response using the retrieved content.
  async function generate(state: typeof MessagesAnnotation.State) {
    // Get recently generated ToolMessages, which will contain the result of Similarity search on vector db.
    let recentToolMessages = [];
    for (let i = state.messages.length - 1; i >= 0; i--) {
      let message = state["messages"][i];
      if (message instanceof ToolMessage) {
        recentToolMessages.push(message);
      } else {
        break;
      }
    }
    let toolMessages = recentToolMessages.reverse();

    // Format into prompt
    const docsContent = toolMessages.map((doc) => doc.content).join("\n");
    const systemMessageContent =
      "You are an assistant for question-answering tasks. " +
      "Use the following pieces of retrieved context to answer " +
      "the question. If you don't know the answer, say that you " +
      "don't know. Use three sentences maximum and keep the " +
      "answer concise." +
      "\n\n" +
      `${docsContent}`;

    /* keep only relevant messages like user and system msg,
    and only add ai-msg if we invoke this without tool calling.
  */
    const conversationMessages = state.messages.filter(
      (message) =>
        message instanceof HumanMessage ||
        message instanceof SystemMessage ||
        (message instanceof AIMessage && message?.tool_calls?.length == 0)
    );
    const prompt = [
      new SystemMessage(systemMessageContent),
      ...conversationMessages,
    ];

    // Run
    const response = await llm.invoke(prompt);
    return { messages: [response] };
  }






  const graphBuilder = new StateGraph(MessagesAnnotation)
    .addNode("queryOrRespond", queryOrRespond)
    .addNode("tools", tools)
    .addNode("generate", generate)
    .addEdge("__start__", "queryOrRespond")
    .addConditionalEdges("queryOrRespond", toolsCondition, {
      __end__: "__end__",
      tools: "tools",
    })
    .addEdge("tools", "generate")
    .addEdge("generate", "__end__");

  const checkpointer = new MemorySaver();
  const graph = graphBuilder.compile({ checkpointer });
  const threadConfig = {
    configurable: { thread_id: "abc123" },
    streamMode: "values" as const,
  };






  // let inputs1 = {
  //   messages: [{ role: "user", content: "What is Task Decomposition?" }],
  // };

  // /**
  //  * "stream":  Instead of waiting until the whole graph finishes, you get live visibility into each node’s output.
  //  * streamMode: "values" means step obj will contain Full state after each graph node executes
  //  */
  // for await (const step of await graph.stream(inputs1, threadConfig)) {
  //   // console.log(step.messages);
  //   const lastMessage = step.messages[step.messages.length - 1];
  //   prettyPrint(lastMessage);
  //   console.log("-----\n");
  // }

  // let inputs2 = {
  //   messages: [
  //     {
  //       role: "user",
  //       content: "Can you look up some common ways of doing it?",
  //     },
  //   ],
  // };

  // for await (const step of await graph.stream(inputs2, threadConfig)) {
  //   const lastMessage = step.messages[step.messages.length - 1];
  //   prettyPrint(lastMessage);
  //   console.log("-----\n");
  // }






  /**
   * Above is one way of doing it. Now if we have multiple query and we want to execute them at once instead of step by step like above we can use agents for that.
   */

  const agent = createReactAgent({ llm: llm, tools: [retrieve] });


  let inputMessage = `What is the standard method for Task Decomposition?
  Once you get the answer, look up common extensions of that method.`;

  let inputs3 = {
    messages: [{
      role: "user", content: inputMessage
    }]
  };

  for await (const step of await agent.stream(inputs3, {
    streamMode: "values",
  })) {
    const lastMessage = step.messages[step.messages.length - 1];
    prettyPrint(lastMessage);
    console.log("-----\n");
  }









}

main();
