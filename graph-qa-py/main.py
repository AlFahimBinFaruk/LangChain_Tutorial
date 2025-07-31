import os
from  dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain.chat_models import init_chat_model
from operator import add
from typing import Annotated,List,Literal
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_neo4j import Neo4jVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from typing import List,Optional




load_dotenv()











#### ************* Configuring the DB *************

NEO4J_URI=os.getenv("NEO4J_URI")
NEO4J_USERNAME=os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD=os.getenv("NEO4J_PASSWORD")

graph=Neo4jGraph(NEO4J_URI,NEO4J_USERNAME,NEO4J_PASSWORD)

movies_query = """
LOAD CSV WITH HEADERS FROM 
'https://raw.githubusercontent.com/tomasonjo/blog-datasets/main/movies/movies_small.csv'
AS row
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
"""

# graph.query(movies_query)

# print(graph.schema)










#### ************* Configuring LLM *************

llm=init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# print(llm.invoke("hello gemini"))







#### ************* Defining States *************

class InputState(TypedDict):
    question:str

class OverAllState(TypedDict):
    question:str
    next_action:str
    cypher_statement:str
    cypher_errors:List[str]
    database_records:List[dict]
    steps:Annotated[List[str],add] # store which step are complated

class OutputState(TypedDict):
    answer:str
    steps:List[str]
    cypher_statement:str







#### ************* Configuring Guard rails *************
#### The purpose of this section is to decide weather the query is related to movie or not.

guardrails_system = """
    As an intelligent assistant, your primary objective is to decide whether a given question is related to movies or not. 
    If the question is related to movies, output "movie". Otherwise, output "end".
    To make this decision, assess the content of the question and determine if it refers to any movie, actor, director, film industry, 
    or related topics. Provide only the specified output: "movie" or "end".
"""

guardrails_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            guardrails_system,
        ),
        (
            "human",
            ("{question}"),
        ),
    ]
)

class GuardrailsOutput(BaseModel):
    decision:Literal["movie","end"]=Field(
        description="Decision on whether the question is related to movies"
    )

guardrails_chain=guardrails_prompt | llm.with_structured_output(GuardrailsOutput)

def guardrails(state:InputState)->OverAllState:
    guardrails_output=guardrails_chain.invoke({"question":state.get("question")})
    database_records=None
    if guardrails_output.decision=="end":
        database_records = "This questions is not about movies or their cast. Therefore I cannot answer this question."
    
    return{
        "next_action":guardrails_output.decision,
        "database_records":database_records,
        "steps":["guardrail"]
    }










### *************** Configuring Few-shot examples ***************

examples = [
    {
        "question": "How many artists are there?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie) RETURN count(DISTINCT a)",
    },
    {
        "question": "Which actors played in the movie Casino?",
        "query": "MATCH (m:Movie {title: 'Casino'})<-[:ACTED_IN]-(a) RETURN a.name",
    },
    {
        "question": "How many movies has Tom Hanks acted in?",
        "query": "MATCH (a:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN count(m)",
    },
    {
        "question": "List all the genres of the movie Schindler's List",
        "query": "MATCH (m:Movie {title: 'Schindler's List'})-[:IN_GENRE]->(g:Genre) RETURN g.name",
    },
    {
        "question": "Which actors have worked in movies from both the comedy and action genres?",
        "query": "MATCH (a:Person)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g1:Genre), (a)-[:ACTED_IN]->(:Movie)-[:IN_GENRE]->(g2:Genre) WHERE g1.name = 'Comedy' AND g2.name = 'Action' RETURN DISTINCT a.name",
    },
    {
        "question": "Which directors have made movies with at least three different actors named 'John'?",
        "query": "MATCH (d:Person)-[:DIRECTED]->(m:Movie)<-[:ACTED_IN]-(a:Person) WHERE a.name STARTS WITH 'John' WITH d, COUNT(DISTINCT a) AS JohnsCount WHERE JohnsCount >= 3 RETURN d.name",
    },
    {
        "question": "Identify movies where directors also played a role in the film.",
        "query": "MATCH (p:Person)-[:DIRECTED]->(m:Movie), (p)-[:ACTED_IN]->(m) RETURN m.title, p.name",
    },
    {
        "question": "Find the actor with the highest number of movies in the database.",
        "query": "MATCH (a:Actor)-[:ACTED_IN]->(m:Movie) RETURN a.name, COUNT(m) AS movieCount ORDER BY movieCount DESC LIMIT 1",
    },
]

example_selector=SemanticSimilarityExampleSelector.from_examples(examples,GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),Neo4jVector,k=5,input_keys=["question"])







### *************** Generating the Cypher ***************

text2cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "Given an input question, convert it to a Cypher query. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """
                You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run.
                Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!
                Here is the schema information
                {schema}

                Below are a number of examples of questions and their corresponding Cypher queries.

                {fewshot_examples}

                User input: {question}
                Cypher query:
                """
            ),
        ),
    ]
)

text2cypher_chain=text2cypher_prompt | llm | StrOutputParser()

def generate_cypher(state:OverAllState)->OverAllState:
    NL="\n"
    blocks=[]
    for example in example_selector.select_examples({"question":state.get("question")}):
        blocks.append(f"Question:{example["question"]}\nCypher:{example['query']}")
    
    fewshot_examples=NL.join(blocks)

    generated_cypher=text2cypher_chain.invoke({
        "question":state.get("question"),
        "fewshot_examples":fewshot_examples,
        "schema":graph.schema
    })

    return {"cypher_statement": generated_cypher, "steps": ["generate_cypher"]}









### *************** Query Validation ***************

validate_cypher_system = """
    You are a Cypher expert reviewing a statement written by a junior developer.
    """

validate_cypher_user = """
    You must check the following:
    * Are there any syntax errors in the Cypher statement?
    * Are there any missing or undefined variables in the Cypher statement?
    * Are any node labels missing from the schema?
    * Are any relationship types missing from the schema?
    * Are any of the properties not included in the schema?
    * Does the Cypher statement include enough information to answer the question?

    Examples of good errors:
    * Label (:Foo) does not exist, did you mean (:Bar)?
    * Property bar does not exist for label Foo, did you mean baz?
    * Relationship FOO does not exist, did you mean FOO_BAR?

    Schema:
    {schema}

    The question is:
    {question}

    The Cypher statement is:
    {cypher}

    Make sure you don't make any mistakes!
"""

validate_cypher_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            validate_cypher_system,
        ),
        (
            "human",
            (validate_cypher_user),
        ),
    ]
)


# defining what data of property that we want to keep.
class Property(BaseModel):
    node_label:str=Field(
        description="The label of the node to which this property belongs."
    )
    property_key: str = Field(
        description="The key of the property being filtered."
    )
    property_value: str = Field(
        description="The value that the property is being matched against."
    )

# defining cypher output after validation
class ValidateCypherOutput(BaseModel):
    errors:Optional[List[str]]=Field(
        description="A list of syntax or semantical errors in the Cypher statement. Always explain the discrepancy between schema and Cypher statement"
    )
    filters:Optional[List[property]]=Field(
        description="A list of property-based filters applied in the Cypher statement."
    )

validate_cypher_chain=validate_cypher_prompt | llm.with_structured_output(ValidateCypherOutput)

# corrected_schema=[]
# for el in enhanced_graph.structured_schema.get("relationships"):
#     corrected_schema.append(Schema(el["start"], el["type"], el["end"]))