# tinychat

**tinychat** is a framework for building and deploying conversational AI systems. It provides three fundamental abstractions that allow engineers to conceptualize complex conversational flows and applications as simple message processing systems:

1. **Message** - Information quanta (the bits)
2. **MessageProcessor** - Information transformations (the channels)
3. **CompositeProcessor** - Information topology (the graph)

Everything else is derived. Databases, APIs, LLMs, CRMs are all processors, while conversations and agents (e. g., chains, graphs, etc.) are composite processors.

![tinychat](tinychat/assets/tinychat.jpg)

## Quickstart

To get started, you can run our quickstart examples and get familiar with the framework. To start, create a virtual environment and install the dependencies.
```bash
python -m venv .venv
source .venv/bin/activate
uv install
```

Then, you can run the ping pong example with the following command:

```bash
make run
```

## Documentation

For detailed documentation, please refer to our [rules](.cursor/rules/tinychat.mdc).