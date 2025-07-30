import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { TokenTextSplitter } from "@langchain/textsplitters";
import { collapseDocs, splitListOfDocs } from "langchain/chains/combine_documents/reduce";
import { Document } from "@langchain/core/documents";
import { StateGraph, Annotation, Send } from "@langchain/langgraph";
import "dotenv/config";




async function main() {

    const pTagSelector = "p";
    const cheerioLoader = new CheerioWebBaseLoader(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        {
            selector: pTagSelector,
        }
    );

    const docs = await cheerioLoader.load();
    // console.log(docs);

    const llm = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 0
    });

    // Summarize Documents.
    const mapPrompt = await pull<ChatPromptTemplate>("rlm/map-prompt");

    // mapPrompt.promptMessages.forEach((message) => {
    //     console.log(message.lc_kwargs.prompt.template);
    // });

    // Summarize summaries.
    const reducePrompt = await pull<ChatPromptTemplate>("rlm/reduce-prompt");

    // reducePrompt.promptMessages.forEach((message) => {
    //     console.log(message.lc_kwargs.prompt.template);
    // });

    const textSplitter = new TokenTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
    });

    const splitDocs = await textSplitter.splitDocuments(docs);


    let tokenMax = 1000;

    async function lengthFunction(documents: any) {
        const tokenCounts = await Promise.all(
            documents.map(async (doc: any) => {
                return llm.getNumTokens(doc.pageContent);
            })
        );

        return tokenCounts.reduce((sum, count) => sum + count, 0);
    };



    const overAllState = Annotation.Root({
        contents: Annotation<string[]>,
        summaries: Annotation<string[]>({
            reducer: (state, update) => state.concat(update)
        }),
        collapsedSummaries: Annotation<Document[]>,
        finalSummary: Annotation<string>
    });

    interface SummaryState {
        content: string
    }

    // This function is responsible for generating summary based on given content.
    const generateSummary = async (state: SummaryState): Promise<{ summaries: string[] }> => {
        const prompt = await mapPrompt.invoke({ docs: state.content });
        const response = await llm.invoke(prompt);
        return { summaries: [String(response.content)] };
    }


    // This will iterate over all the contents all call generateSummary to get the summary.
    const mapSummaries = (state: typeof overAllState.State) => {
        return state.contents.map((content) => new Send("generateSummary", { content }))
    }


    // ****************** If our generated summaries token size is more that what we allocated then we have to summarize them again, and that's what we are going todo in below step's. *********************

    // This function will update our collapsedSummaries value with the summary we already have.
    const collectSummaries = async (state: typeof overAllState.State) => {
        return {
            collapsedSummaries: state.summaries.map(
                (summary) => new Document({ pageContent: summary })
            )
        };
    };


    // This function will call the llm with the prompt to summarize the summary.
    async function _reduce(input: any) {
        const prompt = await reducePrompt.invoke({ doc_summaries: input });
        const response = await llm.invoke(prompt);
        return String(response.content);
    }


    const collapseSummaries = async (state: typeof overAllState.State) => {

        // console.log("collapsed Summaries => ", state.collapsedSummaries);
        // Divides the collapsedSummaries in to array of sub-list each of which length is <= tokenMax.
        const docLists = splitListOfDocs(
            state.collapsedSummaries,
            lengthFunction,
            tokenMax
        );
        // console.log("doclist => ", docLists, "\n");
        const results = [];
        // this will use the pass each summary to _reduce to further summarize them.
        for (const docList of docLists) {
            results.push(await collapseDocs(docList, _reduce))
        };
        // console.log("result => ", results);

        return { collapsedSummaries: results };

    }



    // This represents a conditional edge in the graph that determines
    // if we should collapse the summaries or not
    async function shouldCollapse(state: typeof overAllState.State) {
        let numTokens = await lengthFunction(state.collapsedSummaries);
        if (numTokens > tokenMax) {
            return "collapseSummaries";
        } else {
            return "generateFinalSummary";
        }

    }


    // This function will summarize the summary(with _reduce of course)
    const generateFinalSummary = async (state: typeof overAllState.State) => {
        const response = await _reduce(state.collapsedSummaries);
        return { finalSummary: response };
    };


    const graph = new StateGraph(overAllState)
        .addNode("generateSummary", generateSummary)
        .addNode("collectSummaries", collectSummaries)
        .addNode("collapseSummaries", collapseSummaries)
        .addNode("generateFinalSummary", generateFinalSummary)
        .addConditionalEdges("__start__", mapSummaries, ["generateSummary"])
        .addEdge("generateSummary", "collectSummaries")
        .addConditionalEdges("collectSummaries", shouldCollapse, ["collapseSummaries", "generateFinalSummary"])
        .addConditionalEdges("collapseSummaries", shouldCollapse, ["collapseSummaries", "generateFinalSummary"])
        .addEdge("generateFinalSummary", "__end__");



    const app = graph.compile();


    // Testing it out.
    let finalSummary = null;

    // recursionLimit: 10 means the graph should reach __end__ within 10 super-steps(one iteration where active graph nodes (possibly running in parallel) consume incoming messages, execute their logic, and send messages)
    for await (const step of await app.stream(
        { contents: splitDocs.map((doc) => doc.pageContent) },
        { recursionLimit: 10 }
    )) {
        if (step.hasOwnProperty("generateFinalSummary")) {
            finalSummary = step.generateFinalSummary;
        }
    }

    console.log(finalSummary);

}

main();