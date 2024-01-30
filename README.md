Chat-with-docs-or-url-Chatbot
This Chatbot is an interactive app developed to assist users to interact with their PDF, Docs, txt or urls. It is built using Open Source Stack. No OpenAI is required.

Getting Started
Follow these steps to set up and run the project on your local machine.


#Installation
**Clone the repository**
git clone <repository_url>


**Create the necessary folders**

mkdir db
mkdir docs

**Add your model files to the 'models' folder**
clone LaMini-T5-738M from its git repository
https://huggingface.co/MBZUAI/LaMini-T5-738M


----
### Usage 


**Run the ingestion script to prepare the data** 
`python ingest.py`


**Start the chatbot application using Streamlit** 
`streamlit run chatbot_app.py`
