"""A retrieval step with a custom node (entail product track P5): the prompt built from the retrieved passages must
leave room in the model's context window for the answer. Tokens are counted by words here; a real app counts with its
tokenizer. No model, no GPU.

    ENTAIL=load python examples/custom_nodes/rag_budget.py            # the prompt fits: nothing broken
    ENTAIL=load python examples/custom_nodes/rag_budget.py --fault    # too many passages: broken at rag.prompt
"""
import sys

from entail import nodes

CONTEXT = 256          # the model's context window, in tokens
ANSWER = 64            # tokens kept for the answer
PASSAGES = 12 if "--fault" in sys.argv else 3


@nodes.validator("within_context")
def within_context(tokens, context, reserve):
    n = len(tokens)
    return nodes.ok() if n + reserve <= context else \
        nodes.broken(f"{n} prompt tokens + {reserve} for the answer > the context of {context}")


def retrieve(question, k):
    return [f"passage {i} about {question}: " + "word " * 20 for i in range(k)]


def build_prompt(question):
    text = "\n".join(retrieve(question, PASSAGES)) + f"\nQuestion: {question}\nAnswer:"
    nodes.check("rag.prompt", text.split(), within_context, context=CONTEXT, reserve=ANSWER)
    return text


if __name__ == "__main__":
    prompt = build_prompt("the moon")
    print(len(prompt.split()), "prompt tokens for a", CONTEXT, "token window")
