import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";


const loader = new PDFLoader("/home/bs00927/Downloads/nke-10k-2023.pdf");

const docs = await loader.load();

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,
  chunkOverlap: 200,
});

const allSplits = await textSplitter.splitDocuments(docs);
// console.log("Split: ",allSplits)

const model = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});



const vectorStore = new Chroma(model, {
  collectionName: "a-test-collection",
  url: "http://localhost:8000",
  collectionMetadata: {
    "hnsw:space": "cosine",
  },
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

const filter = { source: "https://example.com" };

const similaritySearchResults = await vectorStore.similaritySearch(
  "biology",
  2,
  filter
);
