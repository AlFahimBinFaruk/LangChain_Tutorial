import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import "dotenv/config";

const model = new ChatGoogleGenerativeAI({
  model: "gemini-2.0-flash",
  temperature: 0,
});


const systemTemplate = "Translate the following text into {language}";

const promptTemplate=ChatPromptTemplate.fromMessages([
    ["system",systemTemplate],
    ["user","{text}"]
])

// const messages = [
//   new SystemMessage(""),
//   new HumanMessage("bike"),
// ];

// const res=await model.invoke(messages);
// console.log("Response => ",res)

const promptValue=await promptTemplate.invoke({
    language:"bangla",
    text:"What is this?"
})

const response = await model.invoke(promptValue);
console.log(" Response is => ",response.content)