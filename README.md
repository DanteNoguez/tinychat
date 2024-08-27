# tinychat

![tinychat](assets/tinychat.jpg)

A simple framework to build reliable conversational agents with AI.

## Quickstart
Clone the repo and set up a virtual environment:
```
git clone https://github.com/DanteNoguez/tinychat
python3 -m venv venvdev
source venvdev/bin/activate
pip install poetry
poetry install
poetry shell
```

Instantiate an .env file with your OpenAI key:
```
OPENAI_API_KEY=
```

You're ready to go, just run the quickstart and start chatting!
```
make quickstart
```

## to-dos
- documentation
- architecture diagrams
- in memory vector db

## Roadmap
- long-term memory, entities memory
- Proxy server (queue, token limits and keys management, load balancing)
- Telemetry, performance monitoring, error tracking
- Support other LLMs and VectorDBs (lazy imports!)
