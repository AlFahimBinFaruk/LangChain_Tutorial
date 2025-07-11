import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChromaClient } from "chromadb";

async function main() {
  const loader = new PDFLoader("/home/bs00927/Downloads/nke-10k-2023.pdf");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 200,
  });
  const allSplits = await splitter.splitDocuments(docs);

  const embeddings = new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2",
  });
  const client = new ChromaClient({
    path: "http://localhost:8000",
    ssl: false,
  });
  const vectorStore = new Chroma(embeddings, {
    client,
    collectionName: "a-test-collection",
    collectionMetadata: { "hnsw:space": "cosine" },
  });

  // console.log("res => ", allSplits);

  // const docRes=await embeddings.embedDocuments(allSplits)
  // console.log(docRes)
  //   const documents = allSplits.map((doc, index) => ({
  //   pageContent: doc.pageContent,
  //   metadata: {
  //     source: doc.metadata.source,
  //     pageNumber: doc.metadata.loc?.pageNumber || null,
  //     // Add other relevant metadata fields here
  //   },
  //   id: String(index + 1), // Ensure unique IDs
  // }));
  //   await vectorStore.addDocuments(documents,{
  //   ids: documents.map((_, index) => String(index + 1)),
  // });

  // const filter = { source: "https://example.com" };

  const embedding = await embeddings.embedQuery(
    "How were Nike's margins impacted in 2023?"
  );

  const results = await vectorStore.similaritySearchVectorWithScore(
    [embedding],
    1
  );

  console.log("Result => ", results);
}

main().catch(console.error);
