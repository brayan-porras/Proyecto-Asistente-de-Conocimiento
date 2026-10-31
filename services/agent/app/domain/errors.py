class AgentError(RuntimeError):
    pass


class LLMError(AgentError):
    pass


class RetrievalError(AgentError):
    pass


class WebSearchError(AgentError):
    pass
