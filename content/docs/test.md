---
date: '2024-11-22T14:28:01+05:30'
draft: false
math: true
title: 'Implementing Word2Vec from scratch (including the backpropogation)'
---
### Python and Numpy based implementation for the Word2Vec (Mikolov et al)
Well vector embeddings play crucial role in deep learning as modern neural nets feed on them. In order to apply neural networks on NLP tasks we need to crunch down the language/words to numbers. There's where word embedding come in play. [link] explains more about this concepts and how they if you are new to the field.

In this blog, I'm jumping straigt into the implementation of Word2Vec and try to replicate the [Mikolov et al, 2014] paper. My intention is to go through the bits and pieces of the algorithms proposed in the paper
$$ x=1 $$
\[
f(x) = x^2 + x + 1\]
# This article is under development