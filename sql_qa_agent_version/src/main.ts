import { StateGraph } from "@langchain/langgraph";
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";
import { Annotation } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { QuerySqlTool } from "langchain/tools/sql";
import { SqlToolkit } from "langchain/agents/toolkits/sql";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { AIMessage, BaseMessage, isAIMessage } from "@langchain/core/messages";
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai";
import { PineconeStore } from "@langchain/pinecone";
import { Pinecone as PineconeClient } from "@pinecone-database/pinecone";
import { createRetrieverTool } from "langchain/tools/retriever";
import { Document } from "@langchain/core/documents";
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

    const dataSource = new DataSource({
        type: "sqlite",
        database: "Chinook.db"
    });

    const db = await SqlDatabase.fromDataSourceParams({
        appDataSource: dataSource
    });

    // const temp = await db.run("SELECT * FROM Artist LIMIT 10;");
    // console.log(temp);

    // Initiating LLM.
    const llm = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 0
    });

    // Initiating the sql toolkit
    const toolkit = new SqlToolkit(db, llm);
    const tools = toolkit.getTools();





    // *******************
    // Retriving Artist name and Album title from DB so that we can perform spell checking.
    // *******************
    async function queryAsList(database: any, query: string): Promise<string[]> {
        const res: Array<{ [key: string]: string }> = JSON.parse(
            await database.run(query)
        )
            .flat()
            .filter((el: any) => el != null);
        const justValues: Array<string> = res.map((item) =>
            Object.values(item)[0]
                .replace(/\b\d+\b/g, "")
                .trim()
        );
        return justValues;
    }

    let artists: string[] = await queryAsList(db, "SELECT Name FROM Artist");
    let albums: string[] = await queryAsList(db, "SELECT Title FROM Album");
    let properNouns = artists.concat(albums);

    // console.log(`Total: ${properNouns.length}\n`);
    // console.log(`Sample: ${properNouns.slice(0, 5)}...`);


    // ******* Setting up embedding model. **********
    const embeddings = new GoogleGenerativeAIEmbeddings({
        model: "text-embedding-004" // 768 dimensions

    });

    // ******* Setting up pinecone db. **********
    const pinecone = new PineconeClient();
    const pineconeIndex = pinecone.Index("sql-qa");

    const vectorStore = new PineconeStore(embeddings, {
        pineconeIndex,
        maxConcurrency: 5,
    });


    const documents = properNouns.map((text) => new Document({ pageContent: text }));

    // ******** adding the documents in vector db- only once ******* .
    await vectorStore.addDocuments(documents);

    // ************** Initializing the retriver tool. ****************
    // give me top 5 results.
    const retriver = vectorStore.asRetriever(5);

    const retriverTool: any = createRetrieverTool(retriver, {
        name: "searchProperNouns",
        description:
            "Use to look up values to filter on. Input is an approximate spelling " +
            "of the proper noun, output is valid proper nouns. Use the noun most " +
            "similar to the search.",
    });

    // const temp = await retriverTool.invoke({ query: "Alice Chains" });
    // console.log(temp);


    // ******* Getting the prompt template for agent. **********
    const systemPromptTemplate = await pull<ChatPromptTemplate>("langchain-ai/sql-agent-system-prompt");

    const systemMessage = await systemPromptTemplate.format({
        dialect: "sqlite",
        top_k: 5,
    });


    // ********** Adding this tool to system tools so our agent can use this. ***********
    let suffix =
        "If you need to filter on a proper noun like a Name, you must ALWAYS first look up " +
        "the filter value using the 'search_proper_nouns' tool! Do not try to " +
        "guess at the proper name - use this function to find similar ones.";

    const system = systemMessage + suffix;

    const updatedTools = tools.concat(retriverTool);

    // Initializing the agent
    const agent = createReactAgent({
        llm: llm,
        tools: updatedTools,
        stateModifier: system
    });

    // ******* Generating ans based on user input. ********
    let inputs2 = {
        messages: [
            { role: "user", content: "How many albums does alis in chain have?" },
        ],
    };

    for await (const step of await agent.stream(inputs2, {
        streamMode: "values",
    })) {
        const lastMessage = step.messages[step.messages.length - 1];
        prettyPrint(lastMessage);
        console.log("-----\n");
    }




}

main();