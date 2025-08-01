import { StateGraph } from "@langchain/langgraph";
import { SqlDatabase } from "langchain/sql_db";
import { DataSource } from "typeorm";
import { Annotation } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { pull } from "langchain/hub";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { QuerySqlTool } from "langchain/tools/sql";
import { z } from "zod";
import "dotenv/config";



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


    const InputStateAnnotation = Annotation.Root({
        question: Annotation<String>
    });

    const StateAnnotation = Annotation.Root({
        question: Annotation<String>,
        query: Annotation<String>,
        result: Annotation<String>,
        answer: Annotation<String>
    });

    // Initiating LLM.
    const llm = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 2
    });

    // Getting the prompt template.
    const queryPromptTemplate = await pull<ChatPromptTemplate>("langchain-ai/sql-query-system-prompt");

    // ******* Generating SQL query based on user input. ********
    const queryOutput = z.object({
        query: z.string().describe("Syntactically valid SQL query.")
    });

    const structuredLLM = llm.withStructuredOutput(queryOutput);

    const writeQuery = async (state: typeof InputStateAnnotation.State) => {
        const promptValue = await queryPromptTemplate.invoke({
            dialect: db.appDataSourceOptions.type,
            top_k: 10,
            table_info: await db.getTableInfo(),
            input: state.question
        });
        const result = await structuredLLM.invoke(promptValue);
        return { query: result.query }
    }

    // const temp = await writeQuery({ question: "How many Employees are there?" });
    // console.log(temp);


    // ************* Executing the query. ************
    const executeQuery = async (state: typeof StateAnnotation.State) => {
        const executeQueryTool = new QuerySqlTool(db);
        console.log("query => ", state.query)
        return { result: await executeQueryTool.invoke(state.query) };
    }

    // const temp = await executeQuery({
    //     question: "",
    //     query: "SELECT COUNT(*) AS EmployeeCount FROM Employee;",
    //     result: "",
    //     answer: "",
    // });
    // console.log(temp);

    // ********* Connecting everything to generate ans **********
    const generateAns = async (state: typeof StateAnnotation.State) => {
        const promptValue =
            "Given the following user question, corresponding SQL query, " +
            "and SQL result, answer the user question.\n\n" +
            `Question: ${state.question}\n` +
            `SQL Query: ${state.query}\n` +
            `SQL Result: ${state.result}\n`;
        const response = await llm.invoke(promptValue);
        return { answer: response.content };
    }

    const graphBuilder = new StateGraph({ stateSchema: StateAnnotation })
        .addNode("writeQuery", writeQuery)
        .addNode("executeQuery", executeQuery)
        .addNode("generateAns", generateAns)
        .addEdge("__start__", "writeQuery")
        .addEdge("writeQuery", "executeQuery")
        .addEdge("executeQuery", "generateAns")
        .addEdge("generateAns", "__end__");

    const graph = graphBuilder.compile();

    let inputs = { question: "Give me employees whose age is >18. Current year is 2025 use employee birth-date to determine age." };

    for await (const step of await graph.stream(inputs, { streamMode: "updates" })) {
        console.log(step);
    }


}

main();