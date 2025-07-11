import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/huggingface_transformers";
import { Chroma } from "@langchain/community/vectorstores/chroma";

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

  const vectorStore = new Chroma(embeddings, {
    collectionName: "a-test-collection",
    url: "http://localhost:8000",
    collectionMetadata: {
      "hnsw:space": "cosine",
    },
  });

  // console.log("res => ", allSplits);
  const documents = allSplits.map((doc, index) => ({
    pageContent: doc.pageContent,
    metadata: {
      source: doc.metadata.source,
      pageNumber: doc.metadata.loc?.pageNumber || null,
    },
    id: String(index + 1),
  }));
  await vectorStore.addDocuments(documents, {
    ids: documents.map((_, index) => String(index + 1)),
  });

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
