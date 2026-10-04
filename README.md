# Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS

# How to run?
### STEPS:

Clone the repository

```bash
git clonehttps://github.com/entbappy/Build-a-Complete-Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n medibot python=3.10 -y
```

```bash
conda activate medibot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone & openai credentials as follows:

```ini
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
OPENAI_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
GROQ_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

The application uses OpenAI's hosted embeddings so Vercel does not need to
bundle PyTorch or a local sentence-transformer model. Run `python store_index.py`
after adding `OPENAI_API_KEY` to create the `medicalchatbot-openai` Pinecone index.

For local development with the existing `medicalchatbot` index, install
`requirements-local.txt` and set these values in `.env`:

```ini
EMBEDDING_PROVIDER=local
PINECONE_INDEX_NAME=medicalchatbot
```


```bash
# run the following command to store embeddings to pinecone
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- GROQ
- Pinecone
