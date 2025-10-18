# Design Patterns in TinyChat

## Overview

TinyChat is a framework for building conversational AI systems using composable message processors. At its core, it provides building blocks that allow engineers to conceptualize complex conversational flows as simple, independent units that communicate through message passing.

The framework is built on three fundamental principles:

1. **Message Processors as Independent Units**: Each processor is a self-contained unit that transforms or routes messages
2. **Markovian State Transitions**: Future states depend only on the present state and current message
3. **Complete Graph Topology**: In composite processors, any processor can communicate with any other processor

## Core Architecture

### The Message Processor: Fundamental Unit

```
┌─────────────────────────────────────┐
│      Message Processor              │
│                                     │
│  ┌─────────┐      ┌──────────┐    │
│  │ Message │─────▶│ Process  │    │
│  │  Input  │      │  Logic   │    │
│  └─────────┘      └────┬─────┘    │
│                        │           │
│                        ▼           │
│                  ┌──────────┐     │
│                  │  Output  │     │
│                  │ Message  │     │
│                  └──────────┘     │
│                                    │
│  Internal State:                  │
│  • Processing metrics             │
│  • Task manager                   │
│  • Observers                      │
└────────────────────────────────────┘
```

A message processor is like a pure function with side effects:
- **Input**: A message carrying information
- **Process**: Transformation logic (the processor's "transfer function")
- **Output**: A new message or None
- **State**: Internal metrics and operational data (not domain state)

### The Composite Processor: Orchestration Layer

```
┌────────────────────────────────────────────────────────────┐
│              Composite Processor                           │
│                                                            │
│    ┌──────────┐         ┌──────────┐         ┌─────────┐ │
│    │Processor │◀───────▶│Processor │◀───────▶│Processor│ │
│    │    A     │         │    B     │         │    C    │ │
│    └─────┬────┘         └─────┬────┘         └────┬────┘ │
│          │                    │                   │       │
│          └────────────────────┼───────────────────┘       │
│                               │                           │
│                               ▼                           │
│                      ┌─────────────────┐                  │
│                      │  Routing Logic  │                  │
│                      └─────────────────┘                  │
│                                                            │
│  Shared Resources:                                        │
│  • Task Manager (one for all)                            │
│  • Observers (broadcast to all)                          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

The composite processor creates a **complete graph** where:
- Every processor can directly call any other processor
- Processors are peers, not hierarchical children
- The topology enables flexible, dynamic routing
- All processors share a single task manager and observer pool

### Message Flow as Markov Process

```
     State S₀              State S₁              State S₂
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Processor A  │    │  Processor B  │    │  Processor C  │
│               │    │               │    │               │
│  Has message  │    │  Has message  │    │  Has message  │
│     m₀        │    │     m₁        │    │     m₂        │
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                    │
        │ Process(m₀)        │ Process(m₁)        │ Process(m₂)
        │                    │                    │
        └───────────▶P(S₁|S₀,m₀)─────────▶P(S₂|S₁,m₁)
                                                   │
                                                   ▼
                                            Terminal State
```

Key properties:
- **Memoryless**: The next state depends only on current state and message
- **Discrete**: State transitions happen at message boundaries
- **Stochastic (optional)**: Routing can be deterministic or probabilistic
- **Information Preserving**: Messages carry forward all necessary context

## Information Theory Perspective

### Messages as Information Carriers

In information theory terms, a message is a **signal** carrying **information** (measured in bits):

```
┌──────────────────────────────────────────────────────┐
│ Message = Signal + Context + Metadata                │
│                                                       │
│ Information Content: I(m) = -log₂(P(m))              │
│ Entropy: H(M) = -Σ P(mᵢ) log₂ P(mᵢ)                 │
└──────────────────────────────────────────────────────┘
```

### Processors as Information Channels

Each processor acts as a **channel** that transforms information:

```
    Input           Channel          Output
   Message          (Processor)      Message
     │                  │               │
     │    H(Input)      │   H(Output)   │
     │       ▼          │       ▼       │
     └─────────────────▶P───────────────┘
                        │
                   I(In : Out)
              (Mutual Information)
```

Properties:
- **Channel Capacity**: Maximum information throughput
- **Noise**: Errors or information loss during processing
- **Mutual Information**: How much input information is preserved in output
- **Entropy Reduction**: Processing reduces uncertainty (makes decisions)

### Complete Graph as Maximum Connectivity

```
         A ◀────────▶ B
         ▲ ╲       ╱ ▲
         │   ╲   ╱   │
         │     ╳     │
         │   ╱   ╲   │
         ▼ ╱       ╲ ▼
         C ◀────────▶ D

Maximum edges = n(n-1)/2 for undirected
              = n(n-1) for directed
```

This topology:
- Maximizes **routing flexibility** (minimum path length = 1)
- Minimizes **information latency** (no intermediary hops required)
- Enables **direct peer-to-peer** communication
- Supports **any routing pattern** (star, ring, chain, mesh, etc.)

---

## Simple Example: Router Pattern

Here's how you might build a simple message router:

```
┌─────────────────────────────────────────────────────┐
│  RouterComposite                                    │
│                                                     │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│    │ History  │    │ LLM      │    │ Validator│   │
│    │ Tracker  │    │ Agent    │    │          │   │
│    └──────────┘    └──────────┘    └──────────┘   │
│                                                     │
│  Routing Logic:                                    │
│  • UserMessage → History → LLM                     │
│  • AIMessage → Validator → History                 │
│  • ErrorMessage → History                          │
└─────────────────────────────────────────────────────┘
```

Each processor is independent and testable. The composite simply orchestrates the flow.

---

## Technical Deep Dive

### Pattern 1: Composite Pattern with Peer Injection

**Classical Composite Pattern:**
```
        Parent
       /   |   \
     Child Child Child
```

**TinyChat's Variant: Peer-Injected Composite**
```
        Composite (Orchestrator)
             │
        ┌────┴────┐
        │ Peers   │
        │  A ◀──▶ B
        │  ▲ ╲  ╱ ▲
        │  │   ╳  │
        │  ▼ ╱  ╲ ▼
        │  C ◀──▶ D
        └─────────┘
```

**Key Difference**: Instead of parent-child hierarchy, we have **peer-to-peer** relationships:

```python
# Classical composite: parent calls children
class ClassicalComposite:
    def process(self, msg):
        for child in self.children:
            child.process(msg)  # Top-down only

# TinyChat: peers call each other
class TinyChatComposite:
    def _inject_peer_references(self):
        for proc in processors:
            for peer in processors:
                if peer != proc:
                    setattr(proc, peer.name, peer)  # Direct reference
                    
    # Now processor A can call B directly:
    # result = await self.processor_b.process(message)
```

**Benefits:**
1. **Flexibility**: Any processor can initiate communication with any other
2. **Locality**: No need to route through parent
3. **Symmetry**: No artificial hierarchy
4. **Testability**: Each processor is independently testable

### Pattern 2: Markov Process with Message-Dependent Transitions

**Standard Markov Chain:**
```
P(Sₜ₊₁ | S₀, S₁, ..., Sₜ) = P(Sₜ₊₁ | Sₜ)
```

**TinyChat Extension:**
```
P(Sₜ₊₁ | Sₜ, mₜ) where mₜ is the message at time t
```

The state transition depends on:
1. **Current processor** (Sₜ)
2. **Current message** (mₜ)

**Implementation:**
```python
class CompositeProcessor:
    async def _process(self, message: Message) -> Optional[Message]:
        # State = which processor we route to
        # Transition function = routing logic based on message
        
        if isinstance(message, UserMessage):
            return await self.history.process(message)  # S_t+1 = history
        elif isinstance(message, AIMessage):
            return await self.validator.process(message)  # S_t+1 = validator
        else:
            return None  # Terminal state
```

**Markov Property Enforcement:**
- Each message carries ALL necessary context (conversation_id, metadata)
- Processors don't maintain conversation state
- State is externalized (in ConversationState, databases, etc.)
- This ensures: **P(next | current, message)** is fully determined

### Pattern 3: Observer Pattern for State Monitoring

```
┌─────────────┐         ┌──────────────┐
│  Processor  │────────▶│   Observer   │
│             │  notify │              │
└─────────────┘         └──────────────┘
       │                       │
       │ State Change          │ Monitor
       ▼                       ▼
┌─────────────┐         ┌──────────────┐
│   Process   │         │   Record     │
│   Message   │         │   Metrics    │
└─────────────┘         └──────────────┘
```

**Two-Level State Tracking:**

1. **Processor State** (operational metrics):
   - Messages received/processed
   - Processing times
   - Error counts
   - Current status

2. **Composite State** (orchestration metrics):
   - Messages routed
   - Routing history
   - Active processors

**Separation of Concerns:**
```python
class ProcessorState:
    """What happened in this processor?"""
    messages_received: int
    messages_processed: int
    avg_processing_time_ms: float
    
class CompositeProcessorState:
    """How are processors communicating?"""
    messages_routed: int
    routing_history: List[tuple[str, str]]
    active_processors: List[str]
```

### Pattern 4: Dependency Injection with Shared Resources

**Resource Sharing Model:**
```
┌────────────────────────────────────────┐
│       Composite Processor              │
│                                        │
│  ┌──────────────────────────────┐     │
│  │    Shared Task Manager       │     │
│  │  (one event loop, one pool)  │     │
│  └────────┬──────────┬──────────┘     │
│           │          │                 │
│    ┌──────▼───┐  ┌──▼────────┐       │
│    │Processor │  │ Processor │       │
│    │    A     │  │     B     │       │
│    └──────────┘  └───────────┘       │
└────────────────────────────────────────┘
```

**Why Share Task Manager?**
1. **Single Event Loop**: All async operations coordinated
2. **Resource Efficiency**: One thread pool, not N
3. **Unified Cancellation**: Stop all tasks in one call
4. **Debugging**: Single place to see all active tasks

**Implementation:**
```python
class CompositeProcessor:
    async def setup(self, setup: ProcessorSetup):
        # Setup self first
        await super().setup(setup)
        
        # Pass shared resources to all sub-processors
        sub_setup = ProcessorSetup(
            task_manager=self._task_manager,  # Shared
            observers=self._observers          # Shared
        )
        
        for processor in self._processors.values():
            await processor.setup(sub_setup)
```

### Pattern 5: Strategy Pattern via Message Processing

Each processor implements the **Strategy Pattern** for message processing:

```
┌─────────────────────────────────────────┐
│      Strategy Interface                 │
│  async def _process(message) -> Message │
└─────────────────────────────────────────┘
            ▲           ▲           ▲
            │           │           │
    ┌───────┴──┐   ┌───┴────┐  ┌───┴─────┐
    │ History  │   │  LLM   │  │ Validator│
    │ Strategy │   │Strategy│  │ Strategy │
    └──────────┘   └────────┘  └──────────┘
```

Each strategy:
- Operates on the same message interface
- Returns the same message type (or None)
- Is hot-swappable
- Can be tested independently

### Pattern 6: Information Flow Semantics

**Message as Information Packet:**
```
┌──────────────────────────────────────────┐
│ Message                                  │
│ ├─ id: str         (identity)           │
│ ├─ timestamp: int  (temporal ordering)  │
│ ├─ content: T      (payload/signal)     │
│ ├─ metadata: dict  (context/noise)      │
│ └─ type: Type      (classification)     │
└──────────────────────────────────────────┘
```

**Information Conservation Law:**
```
I(input) + I(context) = I(output) + I(discarded)

Where:
- I(input): Information in incoming message
- I(context): Information from external sources (state, APIs, etc.)
- I(output): Information in outgoing message
- I(discarded): Information filtered/ignored by processor
```

**Processor Transfer Function:**
```
H(Output | Input) = measure of information added by processor
H(Input | Output) = measure of information preserved from input

Ideal processor: 
- High H(Output | Input): Adds valuable context
- High H(Input | Output): Preserves input information
```

### Pattern 7: State Separation Principle

**Critical Design Decision**: Processors don't hold domain state.

```
❌ ANTI-PATTERN:
class ConversationProcessor:
    def __init__(self):
        self.messages = []  # Holds conversation history
        self.user_context = {}  # Holds user state
        
✅ CORRECT PATTERN:
class ConversationProcessor:
    def __init__(self, state_manager: StateManager):
        self.state = state_manager  # External state store
        
    async def _process(self, message):
        # Fetch state
        history = await self.state.get_history(message.conversation_id)
        # Process
        result = self.process_with_history(message, history)
        # Update state
        await self.state.add_message(result)
        return result
```

**Why External State?**
1. **Markov Property**: Enables true memoryless transitions
2. **Scalability**: State can be distributed/persisted
3. **Testing**: Easy to test with mock state
4. **Debugging**: State changes are explicit and traceable

### Complete System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     Application Layer                          │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │            Composite Processor (Orchestrator)            │ │
│  │                                                          │ │
│  │    ┌─────────┐      ┌─────────┐      ┌─────────┐      │ │
│  │    │Processor│◀────▶│Processor│◀────▶│Processor│      │ │
│  │    │    A    │      │    B    │      │    C    │      │ │
│  │    └────┬────┘      └────┬────┘      └────┬────┘      │ │
│  │         │                │                │            │ │
│  │         └────────────────┼────────────────┘            │ │
│  │                          │                             │ │
│  └──────────────────────────┼─────────────────────────────┘ │
│                             │                               │
└─────────────────────────────┼───────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌───────────────┐
│ Task Manager  │    │    Observers   │    │ State Manager │
│ (async ops)   │    │   (monitoring) │    │   (domain)    │
└───────────────┘    └────────────────┘    └───────────────┘
        ▲                     ▲                     ▲
        │                     │                     │
   Event Loop            Metrics Store         Database/Cache
```

### Summary of Design Principles

1. **Composability**: Build complex systems from simple, independent processors
2. **Markovian Transitions**: Future depends only on present state + message
3. **Complete Graph**: Maximum flexibility in routing and communication
4. **State Separation**: Domain state is external, operational state is internal
5. **Information Preservation**: Messages carry all context needed for processing
6. **Resource Sharing**: Shared task management and observation infrastructure
7. **Peer Injection**: Direct processor-to-processor communication
8. **Observable**: All state changes can be monitored

### When to Use These Patterns

**Use TinyChat patterns when:**
- Building conversational AI with complex routing logic
- Need flexible, dynamic message flow between components
- Want testable, independent processing units
- Require observable system behavior
- Need to reason about information flow

**Consider alternatives when:**
- Simple linear pipeline suffices (use chain of processors)
- Need strict hierarchy (use classical composite pattern)
- State transitions need full history (use FSM with memory)
- Processing is purely functional (use simple functions)

---

## Conclusion

TinyChat provides a framework that treats conversational AI as an **information processing network** where:
- **Messages** are discrete information packets
- **Processors** are stateless transformation functions
- **Composites** orchestrate information flow
- **State** is externalized and explicit
- **Routing** follows Markovian transitions

This design makes complex conversational systems easier to reason about, test, and extend while maintaining mathematical rigor and practical efficiency.

