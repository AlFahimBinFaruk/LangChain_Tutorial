import { Chroma } from "@langchain/community/vectorstores/chroma";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import "@chroma-core/default-embed";
import { ChromaClient } from "chromadb";

const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

const client = new ChromaClient({ path: "http://localhost:8000", ssl: false });
const vectorStore = new Chroma(embeddings, {
  client,
  collectionName: "a-test-collection",
  collectionMetadata: { "hnsw:space": "cosine" },
});

const document1 = {
  pageContent: "The powerhouse of the cell is the mitochondria",
  metadata: { source: "https://example.com" },
};

const document2 = {
  pageContent: "Buildings are made out of brick",
  metadata: { source: "https://example.com" },
};

const document3 = {
  pageContent: "Mitochondria are made out of lipids",
  metadata: { source: "https://example.com" },
};

const document4 = {
  pageContent: "The 2024 Olympics are in Paris",
  metadata: { source: "https://example.com" },
};

const documents = [document1, document2, document3, document4];

await vectorStore.addDocuments(documents, { ids: ["1", "2", "3", "4"] });
await vectorStore.delete({ ids: ["4"] });

// const collection = await client.getCollection({ name: "a-test-collection" });
// // console.log(res);
// const res=await collection.get();
// console.log(res)

const testEmbedding = await embeddings.embedQuery("biology");
// console.log(testEmbedding);
const filter = { source: "https://example.com" };

const similaritySearchResults =
  await vectorStore.similaritySearchVectorWithScore([testEmbedding], 2, filter);
console.log(similaritySearchResults)