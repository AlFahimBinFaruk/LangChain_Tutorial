import { z } from "zod";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import "dotenv/config";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";

async function main() {
  const personSchema = z.object({
    name: z.optional(z.string()).describe("The name of the person"),
    hair_color: z
      .optional(z.string())
      .describe("The color of the person's hair if known"),
    height_in_meters: z
      .optional(z.string())
      .describe("Height measured in meters"),
  });

  const dataSchema = z.object({
    people: z.array(personSchema),
  });

  const promptTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      `You are an expert extraction algorithm.
Only extract relevant information from the text.
If you do not know the value of an attribute asked to extract,
return null for the attribute's value.`,
    ],
    ["human", "{text}"],
  ]);

  const llm = new ChatGoogleGenerativeAI({
    model: "gemini-2.0-flash",
    temperature: 0,
  });

  const structure_llm = llm.withStructuredOutput(dataSchema, {
    name: "list of person",
  });
  const prompt1 = await promptTemplate.invoke({
    text: "My name is Jeff, my hair is gray and i am 6 feet tall. Anna has the same color hair as me.",
  });

  const result1 = await structure_llm.invoke(prompt1);
  console.log(result1);
}

main();
