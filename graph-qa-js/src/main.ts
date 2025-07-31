import "neo4j-driver";
import { Neo4jGraph } from "@langchain/community/graphs/neo4j_graph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { GraphCypherQAChain } from "@langchain/community/chains/graph_qa/cypher";
import "dotenv/config";



const url: any = process.env.NEO4J_URI;
const username: any = process.env.NEO4J_USERNAME;
const password: any = process.env.NEO4J_PASSWORD;

async function main() {
    const graph = await Neo4jGraph.initialize({ url, username, password });

    const moviesQuery = `
        LOAD CSV WITH HEADERS FROM 
        'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
        AS row
        // ***** This line will run for every line in CSV. *****
        MERGE (m:Movie {id:row.movieId})
        SET m.released = date(row.released),
            m.title = row.title,
            m.imdbRating = toFloat(row.imdbRating)
        FOREACH (director in split(row.director, '|') | 
            MERGE (p:Person {name:trim(director)})
            MERGE (p)-[:DIRECTED]->(m))
        FOREACH (actor in split(row.actors, '|') | 
            MERGE (p:Person {name:trim(actor)})
            MERGE (p)-[:ACTED_IN]->(m))
        FOREACH (genre in split(row.genres, '|') | 
            MERGE (g:Genre {name:trim(genre)})
            MERGE (m)-[:IN_GENRE]->(g))
    `;

    // ********** This executes the above query. **********
    // const result = await graph.query(moviesQuery);

    // ********** If there is any change in exixting schema. **********
    // await graph.refreshSchema();

    // console.log(graph.getSchema());

    // **** Initiating the LLM. *****
    const llm = new ChatGoogleGenerativeAI({
        model: "gemini-2.0-flash",
        temperature: 0
    });

    // **** Configuring the LLM with the Graph. ****
    const chain = GraphCypherQAChain.fromLLM({ llm, graph });

    // testing it out.
    const response = await chain.invoke({
        query: "What was the cast of the Casino?",
    });
    console.log(response);
}

main();