import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { ChromaClient } from "chromadb";
import "@chroma-core/default-embed";

async function main() {
  const loader = new PDFLoader("/home/bs00927/Downloads/nke-10k-2023.pdf");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 200 });
  const allSplits = await splitter.splitDocuments(docs);

  const cleanDocs = allSplits.map(d => ({
    pageContent: d.pageContent,
    metadata: { source: d.metadata.source }
  }));

  const embeddings = new HuggingFaceTransformersEmbeddings({
    model: "Xenova/all-MiniLM-L6-v2",
  });

  const client = new ChromaClient({ path: "http://localhost:8000", ssl: false });
  const vectorStore = new Chroma(embeddings, {
    client,
    collectionName: "a-test-collection",
    collectionMetadata: { "hnsw:space": "cosine" },
  });

  const texts = cleanDocs.map(d => d.pageContent);
  const rawVecs = await embeddings.embedDocuments(texts);
  const vectors = rawVecs.map(v => Array.isArray(v) ? v : Array.from(v));

  await vectorStore.addVectors(vectors, cleanDocs);

  const rawQuery = await embeddings.embedQuery("biology");
  const queryVector = Array.isArray(rawQuery) ? rawQuery : Array.from(rawQuery);

  const results = await vectorStore.similaritySearchVectorWithScore(queryVector, 2, { source: "https://example.com" });
  console.log("Results:", results);
}

main().catch(console.error);
