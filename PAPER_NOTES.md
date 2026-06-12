 PAPER_NOTES.md

 Paper Notes: Efficient Estimation of Word Representations in Vector Space

 1. Paper Selected

Title: Efficient Estimation of Word Representations in Vector Space

Authors: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean

Area: Natural Language Processing, Word Embeddings, Representation Learning

 2. What I Understood From the Abstract and Introduction

The paper is trying to solve the problem of how to represent words in a useful numerical form.

Earlier NLP systems often treated words as atomic symbols or IDs. For example, "king", "queen", and "apple" may all just be stored as separate indexes in a vocabulary. This is simple, but it does not capture meaning. The model does not automatically know that "king" and "queen" are related.

The paper argues that words should be represented as continuous vectors. If these vectors are trained properly, words that appear in similar contexts will have similar vectors.

The important idea is:

A word gets its meaning from the words around it.

So instead of manually telling the computer what each word means, the model learns word meaning by looking at context in a large text corpus.

 3. Central Claim of the Paper

The central claim of the paper is:

Simple neural architectures can learn high-quality word vectors much faster than older neural language models, while still capturing meaningful semantic and syntactic relationships between words.

The paper claims that CBOW and Skip-gram work better mainly because they reduce unnecessary computational complexity.

Older neural language models used hidden layers and were expensive to train. The authors remove the costly hidden layer and focus directly on learning word representations from context. Because the models are simpler, they can be trained on much larger datasets, and training on more data improves the quality of the learned vectors.

In simple language:

The paper is saying that a simpler model trained on a lot of text can produce better word vectors than a complex model trained slowly on less text.

 4. Why the Paper Claims This Works

The paper’s reasoning is based on context.

Words that appear in similar contexts usually have related meanings.

For example:

"king" and "queen" may appear near words like royal, palace, crown, prince, kingdom.

Because their contexts overlap, their vectors become closer during training.

The model is not given direct definitions of words. It learns relationships by repeatedly solving prediction tasks:

Given context words, predict the middle word.

or

Given a word, predict nearby words.

After many examples, the embedding vectors start storing useful patterns.

This is why vector arithmetic can sometimes work, such as:

king - man + woman ≈ queen

The model has learned some relationship patterns inside the vector space.

 5. Core Architectures Proposed in the Paper

The paper proposes two main architectures:

1. CBOW: Continuous Bag of Words
2. Skip-gram

 6. CBOW Architecture

CBOW stands for Continuous Bag of Words.

CBOW predicts the current middle word using surrounding context words.

Example sentence:

the cat sat on the mat

If the target word is:

sat

and the window size is 2, then the context words are:

the, cat, on, the

So the training example is:

Input: the, cat, on, the
Output: sat

The model architecture is:

Context word IDs
Embedding layer
Average or sum context embeddings
Linear output layer
Predict target word

CBOW ignores the order of context words. It treats them like a bag of words. This makes it simple and efficient.

 7. Skip-gram Architecture

Skip-gram does the opposite of CBOW.

Instead of using context words to predict the middle word, it uses the middle word to predict surrounding words.

Example:

Input: sat
Output: the, cat, on, the

Skip-gram usually gives better semantic word vectors, especially for rare words, but it is slower because one input word is used to predict multiple context words.

 8. Architecture I Will Implement

For this task, I will implement CBOW.

I chose CBOW because:

1. It is one of the two main architectures proposed in the paper.
2. It is simpler to implement and explain clearly.
3. It directly tests the paper’s core idea that context can be used to learn word vectors.
4. It can run on a small machine with limited compute.
5. It allows honest qualitative evaluation through nearest-neighbor word similarity.

The implementation will not copy any pretrained model. The model will be trained from scratch.

 9. What Exactly Needs To Be Implemented

To test the paper’s claim in a small-scale way, I need to implement:

1. Text preprocessing
2. Vocabulary creation
3. Context-target pair generation
4. Embedding layer
5. CBOW forward pass
6. Loss calculation
7. Backpropagation and training
8. Extraction of learned word vectors
9. Evaluation using nearest-neighbor similarity

The most important part is the embedding layer because that is where the learned word vectors are stored.

 10. Dataset Used in the Original Paper

The original paper used very large datasets.

The paper mentions training on large corpora such as Google News, with billions of tokens. It also reports experiments using hundreds of millions of training words.

This scale is not practical for my implementation because I am working with limited time and compute.

 11. Dataset I Will Use

For my implementation, I will use a smaller text corpus.

The aim is not to match the original paper’s accuracy. The aim is to reproduce the core idea honestly.

My dataset will be used to show that the model can learn basic relationships between words from context.

 12. Evaluation Metric Used in the Paper

The paper evaluates word vectors using semantic and syntactic word relationship tasks.

The idea is to test whether vector arithmetic captures relationships.

Example:

big is to bigger as small is to ?

The model computes:

vector("bigger") - vector("big") + vector("small")

Then it finds the nearest word vector using cosine similarity.

The answer is correct only if the nearest word exactly matches the expected word.

So the main metric is accuracy on semantic and syntactic analogy questions.

 13.Evaluation I Will Use

Since my dataset will be much smaller, I will not claim the same accuracy as the paper.

I will evaluate using:

1. Training loss
2. Nearest-neighbor word similarity
3. Simple qualitative examples
4. Optional small analogy tests if the corpus supports them

Example:

Input word: king

Nearest words may include:

queen, prince, royal, man

If the results are imperfect, I will report that honestly and explain the reason.

 14. Baselines Compared in the Paper

The paper compares CBOW and Skip-gram against earlier word representation methods and neural language models.

Important baselines include:

1. Feedforward Neural Network Language Model
2. Recurrent Neural Network Language Model
3. Previously available word vectors from other researchers
4. Other neural-network based representation methods

The paper shows that CBOW and Skip-gram can achieve strong results with much lower training cost.

 15. My Expected Result

I expect my implementation to show:

1. The training loss decreases over time.
2. Some related words appear closer in vector space.
3. The model learns basic word similarity from context.
4. Results will be weaker than the paper because my corpus is much smaller.

This is expected and honest.

 16. Limitations of My Implementation

My implementation has these limitations:

1. Small dataset
2. Smaller vocabulary
3. Lower embedding dimension
4. Fewer training epochs
5. No large-scale distributed training
6. No full semantic-syntactic benchmark
7. CBOW ignores word order
8. Rare words may not get good vectors

These limitations mean my results cannot be directly compared to the original paper’s full results.

 17. Final Understanding

The paper’s main idea is that useful word meaning can be learned from context.

CBOW and Skip-gram are simple models, but they are powerful because they can be trained efficiently on large text data.

The core insight I am implementing is:

Use nearby words as training signal to learn word embeddings.

My goal is not to reproduce Google-scale accuracy. My goal is to show that I understood the paper, implemented the central CBOW architecture, tested it honestly, and explained the results clearly.
