# agentic_loops
Create a Sample Feature to get to know the agentic loops. Using anime recommendation based on genre and rating.


# Required Packages
pip install fastapi uvicorn requests
pip install groq

# If used ollama
irm https://ollama.com/install.ps1 | iex 
ollama pull llama3.2:3b
ollama


# Run the server
uvicorn main:app --reload


# Define GROQ_API_KEY in the .env


# Futher we will modify this existing feature to support both anime and manga incorportaing agentic graphs
