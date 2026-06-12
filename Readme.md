# Rushil_Shah_Task3
IMPLEMENTING WORD2VEC WITH CBOW IMPLEMENTATION
My implementation reproduces the CBOW method and the analogy-style evaluation idea, but it is not a full-scale reproduction of the paper.
The paper reports 36.1% total accuracy for CBOW trained on 783M words, while my small-scale CBOW model achieved 25.0% Top-1 accuracy on 8 curated analogy questions.
The result is explainably worse and not directly comparable due to dataset size, benchmark size, and compute limitations.


# Word2Vec CBOW Implementation

## Paper Selected

Title: Efficient Estimation of Word Representations in Vector Space

Authors: Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean

This project is based on the Word2Vec paper. The paper proposes two simple and efficient architectures for learning word embeddings:

CBOW: Continuous Bag of Words
Skip-gram

For this implementation, I focused on CBOW.

## Project Aim

The aim of this project is to understand and reproduce the core idea of the Word2Vec paper on a smaller scale.

The original paper trained word vectors on very large datasets containing hundreds of millions to billions of words. Reproducing that exactly is not practical with limited compute and time.

So my aim was not to match the exact paper numbers. My aim was to:

1. Understand the central claim of the paper
2. Implement one core architecture from the paper
3. Train word embeddings from scratch
4. Evaluate the model using the same analogy-style idea used in the paper
5. Report results honestly
6. Analyze why the model succeeds or fails

## Central Claim of the Paper

The central claim of the paper is that simple neural architectures can learn meaningful word vectors efficiently.

Older neural language models were more computationally expensive. The Word2Vec paper shows that simpler models like CBOW and Skip-gram can learn useful word representations with lower training cost, especially when trained on large text corpora.

The key idea is:

Words that appear in similar contexts tend to have related meanings.

For example, words like king and queen may appear around words such as royal, palace, crown, kingdom, and throne. Over training, their vectors can become closer in the embedding space.

## Architecture Implemented

I implemented CBOW, which stands for Continuous Bag of Words.

CBOW predicts the middle word using the surrounding context words.

Example sentence:

the king wears a crown

If the target word is:

wears

and the context window size is 2, then the context words are:

the, king, a, crown

So the training example becomes:

Input: the, king, a, crown
Output: wears

The model learns to predict the target word from its surrounding words.

## Model Flow

The model works as follows:

Context word IDs
Embedding layer
Average of context embeddings
Linear layer
Target word prediction

The embedding layer is the most important part of the model. Initially, word vectors are random. During training, the model updates these vectors so that words appearing in similar contexts become closer.

## Why Only CBOW Was Implemented

The paper proposes both CBOW and Skip-gram, but I implemented only CBOW.

This was a deliberate decision because:

1. CBOW is one of the two main architectures proposed in the paper.
2. It directly tests the paper’s idea that context can be used to learn word meaning.
3. It is simpler to implement and explain clearly.
4. It runs efficiently on a normal machine.
5. The task values understanding, reasoning, and honest analysis more than blindly implementing everything.

Skip-gram is discussed in my notes but not implemented in this version.

## Files in This Repository

train_cbow.py
corpus.txt
eval_questions.txt
PAPER_NOTES.md
README.md
results/training_log.txt
results/nearest_neighbors.txt
results/analogy_evaluation.txt

## Dataset Used

The original paper used very large training corpora such as Google News and other large-scale text datasets.

I did not use the original Google News corpus because it is too large for this small reproduction.

Instead, I used a custom curated corpus inspired by the relationship categories from the paper. The corpus includes repeated examples of:

1. Royal family relationships
2. Gender relationships
3. Country-capital relationships
4. Animal-young relationships
5. Computer science words
6. Student-teacher-school words
7. Comparative/superlative word forms
8. Verb tense patterns
9. Fruit category words

This corpus is not the same as the original paper dataset. It is a smaller controlled dataset created to test whether the CBOW idea can learn basic word relationships.

## Dataset Improvement Experiment

Initially, I trained the model on a very small corpus.

The first small corpus gave weak analogy results:

Top-1 accuracy: 0.0%
Top-3 accuracy: 12.5%

This showed that the model learned some local patterns, but the dataset was too small for better generalization.

After that, I expanded the corpus with more repeated relationship examples and more categories. This improved the final analogy result to:

Top-1 accuracy: 25.0%
Top-3 accuracy: 25.0%

This supports the paper’s idea that more data and better coverage improve the quality of word vectors.

## Training Details

Model: CBOW
Embedding dimension: 80
Window size: 2
Epochs: 200
Optimizer: Adam
Loss function: CrossEntropyLoss
Framework: PyTorch

The model was trained from scratch. No pretrained embeddings were used.

## Training Loss

The training loss decreased significantly:

Epoch 1/200, Loss: 5.0653
Epoch 10/200, Loss: 1.0909
Epoch 20/200, Loss: 0.5516
Epoch 50/200, Loss: 0.2977
Epoch 100/200, Loss: 0.2363
Epoch 150/200, Loss: 0.2225
Epoch 200/200, Loss: 0.2099

This shows that the model learned patterns from the corpus.

The loss did not decrease perfectly at every single checkpoint. For example, it increased slightly between some epochs.
This is normal because the data is shuffled, mini-batches vary, and Adam updates weights batch by batch.
The important point is that the overall trend decreased strongly from 5.0653 to 0.2099.

## Evaluation Metric

The paper evaluates word vectors using semantic and syntactic analogy questions.

The analogy format is:

a is to b as c is to ?

The vector operation is:

vector(b)-vector(a)+vector(c)

Then the model finds the nearest word vector using cosine similarity.

Example:

france is to paris as germany is to ?

Expected answer:

berlin

I implemented the same evaluation idea on a small analogy subset using eval_questions.txt.

## Final Analogy Evaluation Results

Total questions attempted: 8
Skipped questions: 0
Top-1 accuracy: 25.0%
Top-3 accuracy: 25.0%

Correct Top-1 results:

man is to king as woman is to ?
Expected: queen
Predicted Top-3: queen, program, as

woman is to queen as man is to ?
Expected: king
Predicted Top-3: king, swimming, teachers

These two examples show that the model learned some gender and royalty relationships.

## Results Compared to the Paper

The original paper reports results on a much larger semantic-syntactic benchmark and trains using very large datasets.

My result is not directly comparable to the paper’s reported numbers because:

1. The original training data is much larger.
2. The original evaluation set is much larger.
3. The original vectors are trained with far more examples.
4. My evaluation uses only 8 analogy questions.
5. My corpus is custom and small.
6. My model uses a simplified PyTorch CBOW implementation.

However, the evaluation method follows the same idea: vector arithmetic plus nearest-neighbor search using cosine similarity.

So the result should be understood as a small-scale reproduction of the method, not a full reproduction of the original benchmark.

## Nearest Neighbor Evaluation

The model also saves nearest-neighbor results in:

results/nearest_neighbors.txt

Nearest-neighbor evaluation checks which words are closest to a selected word in the learned embedding space.

This helps inspect whether related words moved closer together after training.

## Failure Analysis

Some analogy questions failed even after increasing the corpus size.

Example:

france is to paris as germany is to ?
Expected: berlin
Predicted Top-3: of, city, palace

Example:

cat is to kitten as dog is to ?
Expected: puppy
Predicted Top-3: follows, throne, palace

These failures happened for multiple reasons.

First, the dataset is still very small compared to the original paper. Even though the corpus was expanded, words like berlin, rome, puppy, and kitten still appear far fewer times than they would in a real large corpus.

Second, CBOW averages context words. This makes the model efficient, but it also ignores word order. Because of this, it can learn general similarity but may struggle with precise analogy relationships.

Third, the model can overfit the small corpus. The training loss decreased strongly, but analogy accuracy was only 25%. This means the model learned the training patterns but did not fully generalize to all analogy relationships.

Fourth, some words appear in very similar sentence templates. For example, country-capital examples contain many repeated words like is, the, capital, of, country, and city. These frequent context words can dominate the learned vectors.

# What Worked

The model successfully learned from the corpus because training loss decreased from 5.0653 to 0.2099.

The larger corpus improved analogy accuracy from 0.0% Top-1 to 25.0% Top-1.

The model correctly answered some relationship questions, especially:

man -> king
woman -> queen

and

woman -> queen
man -> king

This shows that the CBOW implementation can learn some meaningful structure from context.

## What Did Not Work Perfectly

The model did not perform well on all analogy types.

Capital-country relationships and animal-young relationships were still weak. This is expected because such relationships need many more examples to become stable in vector space.

The results are explainably worse than the original paper because the implementation uses a much smaller corpus and smaller evaluation set.

# Reference Codebase Note

The original Word2Vec project has a public codebase containing C implementations of CBOW and Skip-gram along with analogy evaluation tools.

I referred to the original codebase only to understand the expected training and evaluation style. My submitted implementation is an independent simplified PyTorch CBOW implementation.

# How to Run

Install PyTorch:

pip install torch

Run the training script:

python train_cbow.py

After running, check the results folder:

results/training_log.txt
results/nearest_neighbors.txt
results/analogy_evaluation.txt

 Output Files

training_log.txt contains the loss values printed during training.

nearest_neighbors.txt contains nearest words for selected test words.

analogy_evaluation.txt contains analogy evaluation results including Top-1 accuracy, Top-3 accuracy, expected answers, and predicted outputs.

 Conclusion

This project implements the CBOW architecture from the Word2Vec paper in PyTorch.

The implementation shows that word embeddings can be learned by predicting a word from its surrounding context. The model learned patterns from the corpus, as shown by the decreasing loss. It also achieved 25.0% Top-1 analogy accuracy on a small custom analogy set after increasing the corpus size.

The results do not match the original paper because the original work used much larger training data, larger evaluation sets, and more compute. However, the project successfully reproduces the core idea of CBOW, runs analogy-style evaluation, compares smaller and larger corpus behaviour, and includes honest failure analysis.
