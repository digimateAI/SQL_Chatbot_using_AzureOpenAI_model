import streamlit as st
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_openai import AzureChatOpenAI
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os
import time
from sqlalchemy import create_engine

@st.cache_resource
def init_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
        temperature=0
    )

@st.cache_resource
def init_db():
    server = os.getenv('DB_SERVER')
    database = os.getenv('DB_NAME')
    username = os.getenv('DB_USERNAME')
    password = os.getenv('DB_PASSWORD')
    driver = os.getenv('DB_DRIVER')
    
    conn_str = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}"
    
    engine = create_engine(
        conn_str,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10
    )
    
    return SQLDatabase(engine=engine)

def clean_sql_query(query: str) -> str:
    """Clean the SQL query by removing markdown and unnecessary formatting"""
    # Remove markdown sql indicators
    query = query.replace('```sql', '').replace('```', '')
    
    # Remove 'SQLQuery:' prefix if present
    query = query.replace('SQLQuery:', '')
    
    # Strip whitespace and ensure proper spacing
    query = query.strip()
    
    return query

def create_chain(llm, db):
    chain_sql = create_sql_query_chain(llm, db)
    execute_query = QuerySQLDataBaseTool(db=db)
    
    answer_prompt = PromptTemplate.from_template(
        """Given the following user question, SQL query, and result, provide a clear and concise answer.
        If there's an error, suggest a corrected query.
        
        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        
        Answer: """
    )
    
    answer = answer_prompt | llm | StrOutputParser()
    
    return chain_sql, execute_query, answer

def main():
    load_dotenv()
    
    st.set_page_config(
        page_title="SQL Database Chat",
        layout="wide"
    )
    
    st.title("Ask any question to your customer database")

    llm = init_llm()
    db = init_db()
    
    chain_sql, execute_query, answer_chain = create_chain(llm, db)
    
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    col1, col2 = st.columns([2, 1])

    with col1:
        user_question = st.text_input("Ask your question:")
        
        if user_question:
            with st.spinner('Processing your question...'):
                start_time = time.time()
                
                try:
                    # Generate SQL query
                    sql_query = chain_sql.invoke({"question": user_question})
                    
                    # Clean the query
                    sql_query = clean_sql_query(sql_query)
                    
                    # Execute query
                    query_result = execute_query.invoke(sql_query)
                    
                    # Generate final answer
                    result = answer_chain.invoke({
                        "question": user_question,
                        "query": sql_query,
                        "result": query_result
                    })
                    
                    response_time = time.time() - start_time
                    
                    # Display SQL Query in a code block
                    st.subheader("Generated SQL Query:")
                    st.code(sql_query, language="sql")
                    
                    # Display the answer
                    st.subheader("Answer:")
                    st.write(result)
                    
                    st.success(f"Response time: {response_time:.2f} seconds")
                    
                    # Store both query and answer in history
                    st.session_state.conversation_history.append({
                        "question": user_question,
                        "sql_query": sql_query,
                        "answer": result
                    })
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    with col2:
        st.subheader("Conversation History")
        for item in reversed(st.session_state.conversation_history[-5:]):
            with st.expander(f"Q: {item['question'][:50]}..."):
                st.write("Question:", item['question'])
                st.write("SQL Query:")
                st.code(item['sql_query'], language="sql")
                st.write("Answer:", item['answer'])

if __name__ == "__main__":
    main()