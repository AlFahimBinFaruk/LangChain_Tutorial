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
from langchain_neo4j.chains.graph_qa.cypher_utils import CypherQueryCorrector,Schema
from neo4j.exceptions import CypherSyntaxError
from langgraph.graph import END,START,StateGraph

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
    filters:Optional[List[Property]]=Field(
        description="A list of property-based filters applied in the Cypher statement."
    )

validate_cypher_chain=validate_cypher_prompt | llm.with_structured_output(ValidateCypherOutput)






### ************ LLM offen strugles to currectly determine relationship directions in generated cypher statements, we can use "CypherQueryCorrector" to correct these directions. ************

corrector_schema=[
    Schema(el["start"],el["type"],el["end"])
    for el in graph.structured_schema.get("relationships")
]

cypher_query_corrector=CypherQueryCorrector(corrector_schema)




### ********** Validating the cypher **********

def validate_cypher(state:OverAllState)->OverAllState:
    errors=[]
    mapping_errors=[]

    try:
        # EXPLAIN detects syntax error in Query.
        graph.query(f"EXPLAIN {state.get("cypher_statement")}")
    except CypherSyntaxError as e:
        errors.append(e.message)

    # use this to correct relationship direction in the statement.
    corrected_cypher=cypher_query_corrector(state.get("cypher_statement"))
    if not corrected_cypher:
        errors.append("The generated Cypher statement doesn't fit the graph schema")
    if not corrected_cypher==state.get("cypher_statement"):
        print("Relationship direction was corrected")

    # Use LLM to find additional potential errors and get the mapping for values
    llm_output=validate_cypher_chain.invoke({
        "question":state.get("question"),
        "schema":graph.schema,
        "cypher":state.get("cypher_statement")
    })

    if llm_output.errors:
        errors.extend(llm_output.errors)
    if llm_output.filters:
        for filter in llm_output.filters:
            if(not [
                prop for prop in graph.structured_schema["node_props"][filter.node_label] if prop["property"]==filter.property_key
            ][0]["type"]=="STRING"):
                continue   
            
            # Verifying if the value exixits in graph
            mapping=graph.query(
                f"MATCH (n:{filter.node_label}) WHERE toLower(n.`{filter.property_key}`)=toLower($value) RETURN 'yes' LIMIT 1",{"value":filter.property_value}
            )
            if not mapping:
                print(
                    f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                )
                mapping_errors.append(
                    f"Missing value mapping for {filter.node_label} on property {filter.property_key} with value {filter.property_value}"
                )
    if mapping_errors:
        next_action="end"
    elif errors:
        next_action="correct_cypher"
    else:
        next_action="execute_cypher"
    
    return {
        "next_action":next_action,
        "cypher_statement":corrected_cypher,
        "cypher_errors":errors,
        "steps": ["validate_cypher"]
    }

    






### ********** The Cypher correction step takes the existing Cypher statement, any identified errors, and the original question to generate a corrected version of the query. **********

correct_cypher_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a Cypher expert reviewing a statement written by a junior developer. "
                "You need to correct the Cypher statement based on the provided errors. No pre-amble."
                "Do not wrap the response in any backticks or anything else. Respond with a Cypher statement only!"
            ),
        ),
        (
            "human",
            (
                """
                    Check for invalid syntax or semantics and return a corrected Cypher statement.

                    Schema:
                    {schema}

                    Note: Do not include any explanations or apologies in your responses.
                    Do not wrap the response in any backticks or anything else.
                    Respond with a Cypher statement only!

                    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.

                    The question is:
                    {question}

                    The Cypher statement is:
                    {cypher}

                    The errors are:
                    {errors}

                    Corrected Cypher statement: 
                """
            ),
        ),
    ]
)

correct_cypher_chain=correct_cypher_prompt | llm | StrOutputParser()

def correct_cypher(state:OverAllState)->OverAllState:
    corrected_cypher=correct_cypher_chain.invoke({
        "question":state.get("question"),
        "errors":state.get("cypher_errors"),
        "cypher":state.get("cypher_statement"),
        "schema":graph.schema
    })
    return {
        "next_action":"validate_cypher",
        "cypher_statement":corrected_cypher,
        "steps":["correct_cypher"]
    }












### ********** Execuing the Query **********

no_results = "I couldn't find any relevant information in the database"


def execute_cypher(state:OverAllState)->OverAllState:
    records=graph.query(state.get("cypher_statement"))
    return{
        "database_records":records if records else no_results,
        "next_action":"end",
        "steps":["execute_cypher"]
    }



### ********** Generating the final ans based on the query **********

generate_final_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant",
        ),
        (
            "human",
            (
                """
                    Use the following results retrieved from a database to provide
                    a succinct, definitive answer to the user's question.

                    Respond as if you are answering the question directly.

                    Results: {results}
                    Question: {question}
                """
            ),
        ),
    ]
)

generate_final_chain=generate_final_prompt | llm | StrOutputParser()

def generate_final_answer(state:OverAllState)->OverAllState:
    final_answer=generate_final_chain.invoke(
        {"question":state.get("question"),"results":state.get("database_records")}
    )
    return {
        "answer":final_answer,
        "steps":["generate_final_answer"]
    }







def guardrails_condition(state:OverAllState)->Literal["generate_cypher","generate_final_answer"]:
    if state.get("next_action")=="end":
        return "generate_final_answer"
    elif state.get("next_action")=="movie":
        return "generate_cypher"
    
def validate_cypher_condition(state:OverAllState)->Literal["generate_final_answer","correct_cypher","execute_cypher"]:
    if state.get("next_action") == "end":
        return "generate_final_answer"
    elif state.get("next_action") == "correct_cypher":
        return "correct_cypher"
    elif state.get("next_action") == "execute_cypher":
        return "execute_cypher"
    




### ********* Initiating the Graph *********

langgraph=StateGraph(OverAllState,input_schema=InputState,output_schema=OutputState)

langgraph.add_node(guardrails)
langgraph.add_node(generate_cypher)
langgraph.add_node(validate_cypher)
langgraph.add_node(correct_cypher)
langgraph.add_node(execute_cypher)
langgraph.add_node(generate_final_answer)

langgraph.add_edge(START,"guardrails")
langgraph.add_conditional_edges(
    "guardrails",
    guardrails_condition
)

langgraph.add_edge("generate_cypher","validate_cypher")
langgraph.add_conditional_edges(
    "validate_cypher",
    validate_cypher_condition
)

langgraph.add_edge("correct_cypher", "validate_cypher")
langgraph.add_edge("execute_cypher","generate_final_answer")
langgraph.add_edge("generate_final_answer",END)

langgraph=langgraph.compile()



res=langgraph.invoke({"question": "Who are the Cast of Casino?"})


print(res)

