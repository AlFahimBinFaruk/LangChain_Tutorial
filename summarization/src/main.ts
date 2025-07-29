import "cheerio";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { pull } from "langchain/hub";
import { TokenTextSplitter } from "@langchain/textsplitters";


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

    const mapPrompt = await pull<ChatPromptTemplate>("rlm/map-prompt");

    const reducePrompt = await pull<ChatPromptTemplate>("rlm/reduce-prompt");

    const textSplitter = new TokenTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 0,
    })

    const splitDocs = await textSplitter.splitDocuments(docs);


}

main();