# Langchain_Demos

Don't forget to add your OpenAI API Key to the .env file. That file is currently named env but you should rename it to .env so it becomes a hidden file

pip install openai pinecone-client langchain streamlit pypdf tiktoken

pip install langchain langchain_experimental

There exist 3 Demos: Thesis Query, CSV Manipulation, and General Python Agent

When running as python files, the ones that use streamlit will prompt you to run it using streamlit run filename.py



About Thesis Query:

You can test hugging face LLM as well here by changing the commented out line from line 31 to 30 in  thesis_query.py

You'll need to make a free account on pinecone's website and copy and paste your API key into the thesis_query.py file on line 46. Or you can alternatively figure out how to use Langchain's built in vector store chroma instead. 
