# Kavak AI
A chatbot server built on top of FastAPI and tinychat, built to handle user requests regarding car purchases via API calls and WhatsApp.

# Quickstart
Install ngrok and create a public URL:
```ngrok http 8081```

Clone the repository:
```
git clone https://github.com/DanteNoguez/tinychat
```

Instantiate an .env file with your OpenAI and Twilio keys:
```
OPENAI_API_KEY=
TWILIO_ACCOUNT_SID=
TWILIO_AUTH_TOKEN=
```

Lastly, run the app:
```
make kavak-run
```

You can now interact with the chatbot by sending POST requests to the `/chat` endpoint of your ngrok URL or by sending WhatsApp messages to the Twilio sandbox number!

# Architecture
## Entry points
- Whatsapp
- API (CRUD for conversations)

## Static default agents
- Main agent (customer support)
    - RAG (Kavak knowledgebase)
    - Car data tool: Pandas agent
    - Financial plans tool

## Unit tests
- Pandas agent accuracy

## To-dos
- Authentication
- Proxy server (queues, rate limiting)