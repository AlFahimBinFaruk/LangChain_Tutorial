import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import {
  StateGraph,
  MessagesAnnotation,
  MemorySaver,
} from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import "dotenv/config";

async function main() {
  const tools = [new TavilySearchResults({ maxResults: 3 })];
  const toolNode = new ToolNode(tools);
  const model = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
  }).bindTools(tools);

  function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
    const lastMessage = messages[messages.length - 1] as AIMessage;

    if (lastMessage.tool_calls?.length) {
      return "tools";
    }
    return "__end__";
  }

  async function callModel(state: typeof MessagesAnnotation.State) {
    const response = await model.invoke(state.messages);
    return { messages: [response] };
  }

  const workflow = new StateGraph(MessagesAnnotation)
    .addNode("agent", callModel)
    .addEdge("__start__", "agent")
    .addNode("tools", toolNode)
    .addEdge("tools", "agent")
    .addConditionalEdges("agent", shouldContinue);

  const config = { configurable: { thread_id: 1 } };
  const memory = new MemorySaver();
  const app = workflow.compile({ checkpointer: memory });

  // Use the agent
  let result;
  result = await app.invoke(
    {
      messages: [
        new HumanMessage("What is the price of small pepsodant in Bangladesh?"),
      ],
    },
    config
  );
  console.log(result.messages[result.messages.length - 1].content);

  result = await app.invoke(
    {
      messages: [new HumanMessage("what about dettol Soap?")],
    },
    config
  );
  console.log(result.messages[result.messages.length - 1].content);
}

main();
