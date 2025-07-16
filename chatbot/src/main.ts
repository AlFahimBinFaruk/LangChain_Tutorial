import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import {
  START,
  END,
  MessagesAnnotation,
  StateGraph,
  MemorySaver,
  Annotation,
} from "@langchain/langgraph";
import { v4 as uuidv4 } from "uuid";
import { ChatPromptTemplate } from "@langchain/core/prompts";

import "dotenv/config";
import { trimMessages, HumanMessage } from "@langchain/core/messages";

async function main() {
  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
  });

  const trimmer = trimMessages({
    maxTokens: 3,
    strategy: "last",
    tokenCounter: (msgs) => msgs.length,
    includeSystem: false,
    allowPartial: false,
    startOn: "human",
  });

  const promptTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
    ],
    ["placeholder", "{messages}"],
  ]);

  const GraphAnnotation = Annotation.Root({
    ...MessagesAnnotation.spec,
    language: Annotation<string>(),
  });

  const callModel = async (state: typeof GraphAnnotation.State) => {
    const trimmedMessages = await trimmer.invoke(state.messages);
    const prompt = await promptTemplate.invoke({
      messages: trimmedMessages,
      language: state.language,
    });
    const response = await llm.invoke(prompt);
    return { messages: response };
  };

  const workFlow = new StateGraph(GraphAnnotation)
    .addNode("model", callModel)
    .addEdge(START, "model")
    .addEdge("model", END);

  const memory = new MemorySaver();
  const app = workFlow.compile({ checkpointer: memory });

  const config1 = { configurable: { thread_id: uuidv4() } };

  let input;
  input = {
    messages: [new HumanMessage("Hi, I am bob?")],
    language: "English",
  };
  await app.invoke(input, config1);
  input = {
    messages: [new HumanMessage("I like vanilla ice cream")],
    language: "English",
  };
  await app.invoke(input, config1);
  input = {
    messages: [new HumanMessage("What Flavour of ice-cream i like?")],
    language: "English",
  };
  const result = await app.invoke(input, config1);
  console.log(result.messages[result.messages.length - 1]);
}

main();
