import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import "dotenv/config";

async function main() {
  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
  });

  const taggingPrompt = ChatPromptTemplate.fromTemplate(`
        Extract the desired information from the following passage.
Only extract the properties mentioned in the 'Classification' function.
Passage:
{input}
`);

  const classificationSchema = z.object({
    sentiment: z
      .enum(["happy", "neutral", "sad"])
      .describe("The sentiment of the text"),
    aggressiveness: z
      .number()
      .int()
      .describe(
        "describes how aggressive the statement is on a scale from 1 to 5. The higher the number the more aggressive"
      ),
    language: z
      .enum(["spanish", "english", "french", "german", "italian"])
      .describe("The language the text is written in"),
  });

  // name param is optional, it just gives more context to the model about what the schema actually represent(extracted data).
  const llmWithStructuredOutput: any = llm.withStructuredOutput(
    classificationSchema,
    {
      name: "extractor",
    }
  );
  const prompt1 = await taggingPrompt.invoke({
    input: "I want to play GTA-5.",
  });
  const result1 = await llmWithStructuredOutput.invoke(prompt1);
  console.log(result1);
  const prompt2 = await taggingPrompt.invoke({
    input: "Estoy muy enojado con vos! Te voy a dar tu merecido!",
  });
  const result2 = await llmWithStructuredOutput.invoke(prompt2);
  console.log(result2);
}

main();
