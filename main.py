import torch
import torch.nn as nn
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import sys
from huggingface_hub import hf_hub_download

from app.retriever import get_retriever
from app.llm import get_llm
from app.reranker import rerank
from app.ingest import run_ingestion

load_dotenv()
if not os.path.exists("./docstore.pkl"):
    print("Knowledge base not found, building it...")
    run_ingestion()


def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

class XLMRobertaClassifier(nn.Module):
    def __init__(self, model_name='xlm-roberta-base', num_classes=3, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.encoder.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask

        x = self.dropout(mean_pooled)
        return self.fc(x)


sys.modules['__main__'].XLMRobertaClassifier = XLMRobertaClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "best_model(2).pth"  
model_path = hf_hub_download("sttr47/email-urgency-xlmr", MODEL_FILENAME)
model_name = "xlm-roberta-base"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
classifier_model = None

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, nn.Module):
            classifier_model = checkpoint
        else:
            classifier_model = XLMRobertaClassifier(model_name=model_name, num_classes=3)
            classifier_model.load_state_dict(checkpoint)
        classifier_model.to(device)
        classifier_model.eval()
        print("Model successfully loaded")
    else:
        print("Error: Model not found")
        
except Exception as e:
    print("Error during loading model:", repr(e))
    tokenizer = None
    classifier_model = None


def classify_email(email_text):

    inputs = tokenizer(
        email_text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=384, 
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = classifier_model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, dim=1)
        
    return predicted_class.item(), confidence.item()

def process_email(email_text):
    label, confidence = classify_email(email_text)
    
    print(f"Label: {label}, Confidence: {confidence}")

    if label == 0:
        return {
            "priority": "Low",
            "response": ""
        }

    elif label in [1, 2]:
        priority_name = "Medium" if label == 1 else "Urgent"
        print(f"Searching...")

        retriever = get_retriever()
        llm = get_llm()

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant which answers questions based on context given below. Answer in concise manner. 
        If you don't know the answer or if the relevant information is not in context, answer with "I don't know" or when in Russian say "я не знаю".

        Context:
        {context}

        Question:
        {question}

        Answer:""")

        rag_chain = prompt | llm | StrOutputParser()

        docs = retriever.invoke(email_text)
        docs = rerank(email_text, docs, top_k=5)
        context = format_docs(docs)

        answer = rag_chain.invoke({
            "context": context,
            "question": email_text
        })

        return {
            "priority": priority_name,
            "response": answer
        }

if __name__ == "__main__":
    print("\nPress 'q' to end")
    
    while True:
        user_input = input("\nMail: ")
        
        if user_input.lower().strip() in ['q', 'exit', 'quit']:
            print("Ended.")
            break
        
        if not user_input.strip():
            continue

        result = process_email(user_input)
        
        print(f"Urgency: {result['priority']}")
        print(f"Answer: {result['response']}")
