<h1 align="center">Bureau Bot 🤖</h1>


## Bureau Bot

Bureau Bot is a document chatbot built using the **Streamlit** framework, the **LlamaIndex** framework, and the **OpenAI API**. Its objective is to help users understand the documents from the *Agentur für Arbeit* by providing them with the information included in the official documents.

![Screenshot](bureau_bot_screenshot.png)


### Installation


To run Bureau Bot, you need **Python 3.10.12**.

You can install the required packages using the command:

```
 pip install -r requirements.txt
```


### Usage

To use Bureau Bot run in the terminal:

```
 chatbot_app.py
```

It will start a Streamlit web application that you can access in your browser at `http://localhost:8501/`.

To use Bureau Bot, you need to enter your OpenAi key.

Bureau Bot will display a chat of all the messages exchanged between the user and the chatbot.

In the Bureau Bot app, you can:
* Ask questions about the documents of the *Agentur für Arbeit*.
* Upload a document (in the formats .docx, .doc, or .pdf) by clicking "*Choose a file from your hard drive*"

Bureau Bot will return the pages and the documents used to create a response, you can download these documents by selecting the title of the file you want to download and clicking "*Download*".

### TODO

To improve the chatbot, we will experiment with:
* Other language models: Llama2, Aleph Alpha, and other pre-trained models (e.g., Hugging Face).
* LangChain’s agents (e.g., web search retriever).
* More experiments with retrieval techniques (e.g., [Karpathy’s SVM-based approach](https://github.com/karpathy/randomfun/blob/master/knn_vs_svm.ipynb) or [Hypothetical Document Embedding](https://arxiv.org/abs/2212.10496)).
* [Finetune the embeddings](https://gpt-index.readthedocs.io/en/stable/examples/finetuning/embeddings/finetune_embedding.html) with an open source LLM.
* Integrate [Weights & Biases](https://wandb.ai/site) to monitor the LLM.

### Credits

Bureau Bot was built by **Daniele Didino** and **Marco Zausch** as a portfolio project for the Data Science Retreat (Berlin). This project was mentored by **Antonio Rueda-Toicen**.
