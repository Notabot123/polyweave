# Reasoning

Differentiable forward chaining over a propositional Horn knowledge base: declare facts
and rules in a `PropKB`, then run a `ForwardChainer` to the deductive closure. The
premise conjunction is a product t-norm (a Pi neuron — see [Logic](logic.md)), so
chaining is differentiable in the facts.

::: polyweave.reasoning
