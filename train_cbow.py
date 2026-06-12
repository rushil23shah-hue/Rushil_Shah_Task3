import os
import re
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim

seed=42
random.seed(seed)
torch.manual_seed(seed)

corpus_path="corpus.txt"
eval_path="eval_questions.txt"
results_dir="results"

window_size=2
embedding_dim=80
epochs=200
learning_rate=0.005
batch_size=16

min_count=1

def load_corpus(path):
    # reads the corpus file from the same folder
    if os.path.exists(path):
        with open(path,"r",encoding="utf-8") as f:
            return f.read()

    # fallback corpus in case corpus.txt is not found
    return """
    king queen prince princess royal palace crown kingdom
    king is a royal man and queen is a royal woman
    prince and princess live in the palace
    the king wears a crown and rules the kingdom
    the queen lives in the royal palace

    man woman boy girl father mother family
    father and mother are part of a family
    boy and girl are young people
    man and woman are adults

    paris france berlin germany rome italy city country capital
    paris is the capital of france
    berlin is the capital of germany
    rome is the capital of italy

    cat dog animal pet kitten puppy
    cat and dog are common pets
    kitten is a young cat
    puppy is a young dog

    computer software hardware code program data algorithm
    students write code and build software
    a program uses data and algorithm logic
    computer hardware runs software
    """

def tokenize(text):
    # keeps only lowercase words
    text=text.lower()
    words=re.findall(r"[a-z]+",text)
    return words

def build_vocab(words,min_count):
    # converts every word to an integer id
    counts=Counter(words)
    vocab=[]

    for word,count in counts.items():
        if count>=min_count:
            vocab.append(word)

    vocab=sorted(vocab)

    word_to_idx={}
    idx_to_word={}

    for i,word in enumerate(vocab):
        word_to_idx[word]=i
        idx_to_word[i]=word

    return word_to_idx,idx_to_word

def make_cbow_data(words,word_to_idx,window):
    # creates context to target pairs for cbow
    data=[]

    for i in range(window,len(words)-window):
        target=words[i]

        if target not in word_to_idx:
            continue

        context=[]

        for j in range(i-window,i+window+1):
            if j==i:
                continue

            w=words[j]

            if w in word_to_idx:
                context.append(word_to_idx[w])

        if len(context)==2*window:
            data.append((context,word_to_idx[target]))

    return data

class CBOW(nn.Module):
    # context words are converted into vectors
    # their average is used to predict the target word

    def __init__(self,vocab_size,embedding_dim):
        super(CBOW,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embedding_dim)
        self.output=nn.Linear(embedding_dim,vocab_size)

    def forward(self,context):
        emb=self.embedding(context)
        avg=torch.mean(emb,dim=1)
        out=self.output(avg)
        return out

def train(model,data):
    loss_fn=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=learning_rate)
    logs=[]

    for ep in range(1,epochs+1):
        random.shuffle(data)
        total_loss=0
        batches=0

        for i in range(0,len(data),batch_size):
            batch=data[i:i+batch_size]

            x=[item[0] for item in batch]
            y=[item[1] for item in batch]

            x=torch.tensor(x,dtype=torch.long)
            y=torch.tensor(y,dtype=torch.long)

            optimizer.zero_grad()
            pred=model(x)
            loss=loss_fn(pred,y)

            loss.backward()
            optimizer.step()

            total_loss+=loss.item()
            batches+=1

        avg_loss=total_loss/batches

        if ep==1 or ep%10==0:
            line=f"Epoch {ep}/{epochs}, Loss: {avg_loss:.4f}"
            print(line)
            logs.append(line)

    return logs

def nearest_words(word,word_to_idx,idx_to_word,embeddings,top_k=5):
    # finds words whose vectors are closest to the given word
    if word not in word_to_idx:
        return []

    idx=word_to_idx[word]
    norm_emb=embeddings/embeddings.norm(dim=1,keepdim=True)

    query=norm_emb[idx]
    scores=torch.mv(norm_emb,query)
    scores[idx]=-1

    values,indices=torch.topk(scores,k=min(top_k,len(idx_to_word)-1))

    ans=[]

    for val,ind in zip(values,indices):
        ans.append((idx_to_word[ind.item()],val.item()))

    return ans

def analogy(a,b,c,word_to_idx,idx_to_word,embeddings,top_k=3):
    # simple test like man to king as woman to ?
    for w in [a,b,c]:
        if w not in word_to_idx:
            return []

    norm_emb=embeddings/embeddings.norm(dim=1,keepdim=True)

    va=norm_emb[word_to_idx[a]]
    vb=norm_emb[word_to_idx[b]]
    vc=norm_emb[word_to_idx[c]]

    query=vb-va+vc
    scores=torch.mv(norm_emb,query)

    for w in [a,b,c]:
        scores[word_to_idx[w]]=-1

    values,indices=torch.topk(scores,k=min(top_k,len(idx_to_word)-3))

    ans=[]

    for val,ind in zip(values,indices):
        ans.append((idx_to_word[ind.item()],val.item()))

    return ans
def load_eval_questions(path):
    questions=[]

    if not os.path.exists(path):
        return questions

    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip().lower()

            if line=="" or line.startswith(":"):
                continue

            parts=line.split()

            if len(parts)==4:
                questions.append(tuple(parts))

    return questions

def predict_analogy(a,b,c,word_to_idx,idx_to_word,embeddings,top_k=3):
    for w in [a,b,c]:
        if w not in word_to_idx:
            return []

    norm_emb=embeddings/embeddings.norm(dim=1,keepdim=True)

    va=norm_emb[word_to_idx[a]]
    vb=norm_emb[word_to_idx[b]]
    vc=norm_emb[word_to_idx[c]]

    query=vb-va+vc
    scores=torch.mv(norm_emb,query)

    for w in [a,b,c]:
        scores[word_to_idx[w]]=-1

    values,indices=torch.topk(scores,k=min(top_k,len(idx_to_word)-3))

    ans=[]

    for val,ind in zip(values,indices):
        ans.append(idx_to_word[ind.item()])

    return ans

def evaluate_analogies(word_to_idx,idx_to_word,embeddings):
    questions=load_eval_questions(eval_path)

    total=0
    skipped=0
    top1_correct=0
    top3_correct=0
    logs=[]

    for a,b,c,expected in questions:
        required=[a,b,c,expected]
        missing=False

        for w in required:
            if w not in word_to_idx:
                missing=True
                break

        if missing:
            skipped+=1
            logs.append(f"{a} {b} {c} {expected} -> skipped because word missing")
            continue

        total+=1
        preds=predict_analogy(a,b,c,word_to_idx,idx_to_word,embeddings,top_k=3)

        if len(preds)>0 and preds[0]==expected:
            top1_correct+=1

        if expected in preds:
            top3_correct+=1

        logs.append(f"{a} is to {b} as {c} is to ?")
        logs.append("Expected: "+expected)
        logs.append("Predicted Top-3: "+", ".join(preds))
        logs.append("")

    top1_acc=0
    top3_acc=0

    if total>0:
        top1_acc=(top1_correct/total)*100
        top3_acc=(top3_correct/total)*100

    summary=[]
    summary.append("Analogy Evaluation")
    summary.append("")
    summary.append("Total questions attempted: "+str(total))
    summary.append("Skipped questions: "+str(skipped))
    summary.append("Top-1 accuracy: "+str(round(top1_acc,2))+"%")
    summary.append("Top-3 accuracy: "+str(round(top3_acc,2))+"%")
    summary.append("")
    summary.extend(logs)

    return summary
def save_results(logs,model,word_to_idx,idx_to_word):
    os.makedirs(results_dir,exist_ok=True)

    with open(os.path.join(results_dir,"training_log.txt"),"w",encoding="utf-8") as f:
        for line in logs:
            f.write(line+"\n")

    embeddings=model.embedding.weight.data
    eval_logs=evaluate_analogies(word_to_idx,idx_to_word,embeddings)

    with open(os.path.join(results_dir,"analogy_evaluation.txt"),"w",encoding="utf-8") as f:
        for line in eval_logs:
            f.write(line+"\n")

    test_words=[
        "king","queen","man","woman",
        "paris","france","cat","dog",
        "computer","software","student"
    ]

    with open(os.path.join(results_dir,"nearest_neighbors.txt"),"w",encoding="utf-8") as f:
        f.write("Nearest Neighbor Results\n\n")

        for word in test_words:
            f.write("Word: "+word+"\n")
            near=nearest_words(word,word_to_idx,idx_to_word,embeddings)

            if not near:
                f.write("Not found in vocabulary\n\n")
                continue

            for n,score in near:
                f.write(n+": "+str(round(score,4))+"\n")

            f.write("\n")

        f.write("Analogy Tests\n\n")

        tests=[
            ("man","king","woman"),
            ("france","paris","germany"),
            ("cat","kitten","dog")
        ]

        for a,b,c in tests:
            f.write(a+" is to "+b+" as "+c+" is to ?\n")
            res=analogy(a,b,c,word_to_idx,idx_to_word,embeddings)

            if not res:
                f.write("Could not run this test because some word is missing.\n\n")
                continue

            for word,score in res:
                f.write(word+": "+str(round(score,4))+"\n")

            f.write("\n")

def main():
    print("Loading corpus...")
    text=load_corpus(corpus_path)

    print("Tokenizing text...")
    words=tokenize(text)
    print("Total tokens:",len(words))

    print("Building vocabulary...")
    word_to_idx,idx_to_word=build_vocab(words,min_count)
    print("Vocabulary size:",len(word_to_idx))

    print("Creating CBOW training data...")
    data=make_cbow_data(words,word_to_idx,window_size)
    print("Training examples:",len(data))

    if len(data)==0:
        print("No training data found. Add more text to corpus.txt")
        return

    print("Creating model...")
    model=CBOW(len(word_to_idx),embedding_dim)

    print("Training started...\n")
    logs=train(model,data)

    print("\nSaving results...")
    save_results(logs,model,word_to_idx,idx_to_word)

    print("\nDone.")
    print("Check results/training_log.txt")
    print("Check results/nearest_neighbors.txt")

if __name__=="__main__":
    main()