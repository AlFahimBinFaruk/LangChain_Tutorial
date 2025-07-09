import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";

const loader = new PDFLoader("/home/bs00927/Downloads/nke-10k-2023.pdf");

const docs = await loader.load();
console.log(docs[0].metadata);

const textSplitter=new RecursiveCharacterTextSplitter({
    chunkSize:1000,
    chunkOverlap:200,
});

const allSplits=await textSplitter.splitDocuments(docs);

console.log("Split length => ",allSplits.length)