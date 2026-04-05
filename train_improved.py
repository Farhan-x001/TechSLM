#!/usr/bin/env python3
"""
IMPROVED TechSLM Training Script
- Word-level tokenization (instead of character-level)
- Larger architecture: n_embd=256, n_layer=6, n_head=8
- Larger context window: block_size=256
- Better training data with coherent paragraphs
- More epochs and improved hyperparameters
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import re
import time
from collections import Counter


# ============================================================================
# IMPROVED TRAINING DATA - Coherent paragraphs
# ============================================================================

TRAINING_CORPUS = """
Artificial intelligence and machine learning are transforming every aspect of modern computing. Deep learning models trained on GPT and BERT architectures have demonstrated remarkable capabilities in natural language understanding and generation. The transformer architecture, introduced in the "Attention is All You Need" paper, revolutionized how we process sequential data by enabling parallel computation and capturing long-range dependencies.

Neural networks consist of interconnected layers of neurons that process information through learned weights and biases. Convolutional neural networks excel at image recognition tasks by applying learned filters across spatial dimensions. Recurrent neural networks process sequential data by maintaining hidden state across time steps. The LSTM unit addresses the vanishing gradient problem by using gating mechanisms to control information flow.

Training neural networks requires careful optimization of hyperparameters like learning rate, batch size, and regularization strength. Stochastic gradient descent and its variants like Adam are fundamental optimization algorithms that update weights based on gradient estimates. Momentum accumulates past gradients to accelerate convergence in the relevant direction. Learning rate scheduling adjusts the step size during training to balance convergence speed and final accuracy.

Transfer learning enables knowledge reuse by starting with pretrained weights and fine-tuning on new tasks. Large language models like GPT-3 and GPT-4 demonstrate that scaling up data and parameters leads to emergent capabilities. Few-shot learning allows models to adapt to new tasks with minimal examples. In-context learning enables models to perform tasks described in natural language prompts without explicit fine-tuning.

Computer vision applications span from image classification and object detection to semantic segmentation and instance segmentation. Convolutional architectures like ResNet and EfficientNet achieve state-of-the-art accuracy on benchmark datasets. Vision transformers apply the transformer architecture to images by treating them as sequences of patches. CLIP learns aligned representations of images and text through contrastive learning on large-scale datasets.

Natural language processing encompasses tokenization, part-of-speech tagging, named entity recognition, and sentiment analysis. Language models predict the next word in a sequence based on previous context. Attention mechanisms enable models to weight different parts of the input when producing each output. Multi-head attention applies multiple independent attention operations in parallel to capture diverse relationships.

Recurrent architectures process variable-length sequences by maintaining hidden state. Bidirectional RNNs process sequences in both directions to capture context from future tokens. Sequence-to-sequence models with encoder-decoder architecture enable machine translation and summarization. Attention mechanisms allow decoders to focus on relevant parts of the encoder output when generating each token.

Reinforcement learning trains agents to maximize cumulative reward through interaction with environments. Policy gradient methods directly optimize the policy by computing gradients of expected reward. Value-based methods learn value functions that estimate future cumulative rewards. Actor-critic methods combine policy and value learning for improved convergence and sample efficiency.

Computer architecture encompasses processors, memory hierarchies, and interconnects. CPU cores execute instructions sequentially with sophisticated branch prediction and out-of-order execution. GPU cores execute thousands of threads in parallel to achieve high throughput. Apple's M1 chip integrates CPU and GPU cores on the same die with unified memory architecture. Metal Performance Shaders accelerate computation on Apple Silicon devices through optimized GPU kernels.

Distributed systems coordinate computation across multiple machines through networking and consensus protocols. MapReduce and Spark enable distributed data processing on commodity hardware clusters. Kubernetes orchestrates containerized workloads across clusters, handling scheduling and resource management. Microservices architecture decomposes applications into independently deployable services that communicate through APIs.

Database systems manage persistent storage of structured data with ACID guarantees. Relational databases organize data into tables with schemas and enforce referential integrity. NoSQL databases provide flexible schemas and horizontal scalability for semi-structured data. Data warehouses optimize for analytical workloads using columnar storage and compression. Graph databases efficiently represent and query relationships between entities.

Cryptography provides confidentiality, integrity, and authentication for secure communication. Symmetric encryption like AES uses the same key for encryption and decryption. Public-key cryptography uses key pairs where private keys decrypt messages encrypted with public keys. Hash functions produce fixed-size digests that are computationally difficult to reverse. Digital signatures prove message authenticity and non-repudiation.

Cloud computing provides on-demand access to computing resources through the internet. Infrastructure as a Service provides virtualized computing resources that users can provision and manage. Platform as a Service abstracts away infrastructure management, allowing developers to focus on applications. Software as a Service delivers applications through web browsers without local installation. Edge computing brings computation closer to data sources to reduce latency.

Software engineering practices ensure code quality, maintainability, and reliability. Version control systems like Git track changes and enable collaboration among developers. Continuous integration automatically builds and tests code changes to catch issues early. Continuous deployment automates the process of releasing tested code to production. Automated testing including unit tests, integration tests, and end-to-end tests verifies correctness.

DevOps bridges the gap between development and operations through automation and collaboration. Infrastructure as code version-controls infrastructure configuration, enabling reproducible deployments. Containerization using Docker packages applications and dependencies for consistent deployment across environments. Monitoring and observability track system health and diagnose issues in production. Incident response processes minimize downtime when failures occur.

APIs define interfaces for communication between software components and systems. REST APIs use HTTP methods to access resources identified by URLs. GraphQL provides clients with precise control over requested data fields. gRPC uses Protocol Buffers and HTTP/2 for efficient inter-service communication. Message queues enable asynchronous communication between services.

Web development creates interactive applications accessible through web browsers. Frontend frameworks like React, Vue, and Angular manage dynamic user interfaces efficiently. Backend frameworks like Django and Flask provide tools for building server-side logic. Full-stack development involves building both frontend and backend components. Progressive web apps provide offline functionality and installability like native applications.

Mobile development creates applications for smartphones and tablets. Native development uses platform-specific languages and SDKs for optimal performance. Cross-platform frameworks like React Native and Flutter share code between iOS and Android. Mobile-first design prioritizes small screen sizes and touch interfaces. App store distribution through iOS App Store and Google Play enables monetization.

Cybersecurity protects systems and data from unauthorized access and malicious attacks. Authentication verifies user identity through passwords, tokens, or biometric factors. Authorization determines what authenticated users can access. Firewalls filter traffic based on rules to prevent unauthorized connections. Intrusion detection systems monitor for suspicious activity patterns.

Data science extracts insights from data through statistical analysis and machine learning. Data preprocessing cleans and transforms raw data for analysis. Feature engineering creates meaningful features that improve model performance. Model selection compares different algorithms to find the best approach. Cross-validation estimates model performance on unseen data.

Business intelligence converts raw data into actionable insights for decision making. Data warehouses integrate data from multiple sources into a central repository. Business analytics tools visualize data and enable ad-hoc exploration. Dashboards display key metrics and KPIs in real-time. Data-driven decision making relies on evidence rather than intuition.

Ethics in computing addresses societal impacts of technology. Privacy protection ensures personal information is handled responsibly. Fairness in machine learning prevents discrimination against protected groups. Transparency enables users to understand how systems work. Accountability establishes responsibility for system behavior and impacts.
"""


# ============================================================================
# WORD-LEVEL TOKENIZER
# ============================================================================

class WordTokenizer:
    """Simple word-level tokenizer with special tokens."""
    
    def __init__(self, text):
        # Split into words and build vocabulary
        words = re.findall(r'\b\w+\b|[.,!?;:]', text.lower())
        self.vocab = ['<pad>', '<unk>', '<start>', '<end>'] + sorted(list(set(words)))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx_to_word = {i: w for i, w in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text):
        """Convert text to token indices."""
        words = re.findall(r'\b\w+\b|[.,!?;:]', text.lower())
        return [self.word_to_idx.get(w, 1) for w in words]  # 1 is <unk>
    
    def decode(self, indices):
        """Convert token indices back to text."""
        words = [self.idx_to_word.get(i, '<unk>') for i in indices]
        return ' '.join(words)


class TextDataset(Dataset):
    """Dataset for word-level text generation."""
    
    def __init__(self, text, tokenizer, block_size=256):
        self.tokens = tokenizer.encode(text)
        self.block_size = block_size
    
    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx:idx + self.block_size], dtype=torch.long)
        y = torch.tensor(self.tokens[idx + 1:idx + self.block_size + 1], dtype=torch.long)
        return x, y


# ============================================================================
# IMPROVED MODEL ARCHITECTURE
# ============================================================================

class ImprovedMiniTransformer(nn.Module):
    """Enhanced transformer with larger capacity."""
    
    def __init__(self, vocab_size, n_embd=256, n_head=8, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=n_embd,
                nhead=n_head,
                dim_feedforward=4 * n_embd,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            )
            for _ in range(n_layer)
        ])
        
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        B, T = x.shape
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb(torch.arange(T, device=x.device))
        
        x = token_emb + pos_emb
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        for block in self.transformer_blocks:
            x = block(x, src_key_padding_mask=None, is_causal=True, src_mask=causal_mask)
        
        x = self.ln(x)
        logits = self.lm_head(x)
        return logits


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_improved_model():
    """Train the improved model with word-level tokenization."""
    
    print("\n" + "=" * 70)
    print("IMPROVED TechSLM TRAINING - Word-Level Tokenization")
    print("=" * 70)
    
    # Setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Device: {device}")
    
    # Data preparation
    print(f"\n✓ Building tokenizer...")
    tokenizer = WordTokenizer(TRAINING_CORPUS)
    print(f"  Vocabulary size: {tokenizer.vocab_size}")
    
    # Repeat corpus for more training data
    full_text = TRAINING_CORPUS * 8
    dataset = TextDataset(full_text, tokenizer, block_size=256)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    print(f"  Dataset samples: {len(dataset):,}")
    print(f"  Batches per epoch: {len(dataloader)}")
    
    # Model
    print(f"\n✓ Creating improved model...")
    model = ImprovedMiniTransformer(
        vocab_size=tokenizer.vocab_size,
        n_embd=256,
        n_head=8,
        n_layer=6,
        block_size=256,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50
    
    print(f"\n" + "=" * 70)
    print(f"TRAINING (50 epochs, {len(dataloader)} batches/epoch)")
    print("=" * 70 + "\n")
    
    losses = []
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        model.train()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            B, T, V = logits.shape
            loss = criterion(logits.view(B * T, V), y.view(B * T))
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'tokenizer_vocab': tokenizer.vocab,
                'config': {
                    'vocab_size': tokenizer.vocab_size,
                    'n_embd': 256,
                    'n_head': 8,
                    'n_layer': 6,
                    'block_size': 256,
                },
            }
            torch.save(checkpoint, 'tech_slm_model.pth')
            print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {avg_loss:.4f} ⭐ SAVED")
        else:
            print(f"Epoch {epoch:2d}/{num_epochs} | Loss: {avg_loss:.4f}")
    
    print(f"\n" + "=" * 70)
    print(f"Training complete! Final loss: {losses[-1]:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    train_improved_model()
