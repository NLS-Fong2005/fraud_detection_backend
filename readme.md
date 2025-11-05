LLM_RAG Method

**Frameworks**
1. Langchain
2. Python FastAPI
3. ChromaDB

**Functions**
1. RAG with ChromaDB
   1. Loaded with Documents pertaining to Guidelines
   2. Model's judges row based on:
      1. Message Content
      2. Network Data
      3. Temporal Data
      4. Geographical Data
2. Extract Row by Row
   1. Only judges each column with boolean value
   2. Based on how many TRUE statements, make a percentage

      (e.g. 3 TRUES + 1 FALSE = 75% suspicious. 0 TRUES + 4 FALSE = 100% Suspicious)
   3. Goes column by column left to right, one row at a time. Skips if info is unavailable.