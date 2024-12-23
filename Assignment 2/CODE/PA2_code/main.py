
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import SimpleTokenizer
import nltk
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import Encoder,Decoder
from transformer_3a import Encoder_3a,Decoder_3a
from transformer_3b import Encoder_3b,Decoder_3b
import torch.optim as optim
seed = 42
from utilities import Utilities
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
import argparse



batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.
d_model = 64 # embedding dimension
sequence_length = 32
n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
hidden_size = n_hidden
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(encoder, data_loader):
    encoder.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = encoder(xb, classify=True) 
            _, predicted = torch.max(logits.data, 1)
            total += yb.size(0)
            correct += (predicted == yb).sum().item()
    encoder.train()
    return 100 * correct / total
def compute_perplexity(decoder, data_loader, eval_iters=100):
    decoder.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i, (xb, yb) in enumerate(data_loader):
            if i >= eval_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
      
            decoder_output, _ = decoder(xb, yb)
            loss = F.cross_entropy(decoder_output.view(-1, decoder_output.size(-1)), yb.view(-1))
            total_loss += loss.item()
            n_batches += 1
    decoder.train()
    return torch.exp(torch.tensor(total_loss / n_batches)).item()


def main():
    print("please wait for 1-2 min for punkt_tab to download then start running main file")
    nltk.download('punkt_tab')
    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts))
    print("Vocabulary size is", tokenizer.vocab_size)

    # Load datasets
    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size, 
                                collate_fn=collate_batch, shuffle=True)
    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, 
                                collate_fn=collate_batch, shuffle=True)
 
    
    

   
    parser = argparse.ArgumentParser(description='Your program description')
    parser.add_argument('--part', choices=['part1', 'part2','part3_a','part3_b'], help='Specify which part to run')
    args = parser.parse_args()

    if args.part == 'part1':
        x = 0
    elif args.part == 'part2':
        x = 1
    elif args.part == 'part3_a':
        x = 2
    elif args.part == 'part3_b':
        x = 3
    #Training Classification Task
    if x == 0:
        print("\nTraining Classification Task...")
        encoder = Encoder(vocab_size=tokenizer.vocab_size+1, d_model=d_model, 
                     sequence_length=sequence_length, num_heads=n_head, 
                     num_layers=n_layer, hidden_size=hidden_size).to(device)
        optimizer_cls = optim.Adam(encoder.parameters(), lr=learning_rate)
        utils = Utilities(tokenizer, encoder) #remove
        for epoch in range(epochs_CLS):
            total_loss = 0
            for i, (xb, yb) in enumerate(train_CLS_loader):
                xb, yb = xb.to(device), yb.to(device)
                
                # Forward pass
                logits, loss, _ = encoder(xb, classify=True, targets=yb)
                
                # Backward pass
                optimizer_cls.zero_grad()
                loss.backward()
                optimizer_cls.step()
                
                total_loss += loss.item()
                
                if (i + 1) % eval_interval == 0:
                    avg_loss = total_loss / eval_interval
                    print(f'Classification Epoch [{epoch+1}/{epochs_CLS}], Step [{i+1}], Avg Loss: {avg_loss:.4f}')
                    total_loss = 0

            
            train_accuracy = compute_classifier_accuracy(encoder, train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(encoder, test_CLS_loader)

            print(f"Epoch {epoch+1}/{epochs_CLS}, Train classifier Accuracy: {train_accuracy:.2f}%, Test classifier Accuracy: {test_accuracy:.2f}%")
            # Sanity check
        utilities = Utilities(tokenizer,  encoder)
        test_sentence = "Let me tell you, you're not the only ones."
        utilities.sanity_check(test_sentence, block_size)
        utilities = Utilities(tokenizer,  encoder)
        # test_sentence = "One issue Look at the world on this bright August night."
        # utilities.sanity_check(test_sentence, block_size)
        num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"Number of parameters in the encoder: {num_params}")
        torch.save(encoder.state_dict(), 'encoder_model.pth')
        print("\nModels saved successfully!")
        # test_sentence_2 = "Twenty million more Americans know the financial security of health insurance."
        # test_sentence = "This afternoon, I spoke to former President George W. Bush."
        # utilities.sanity_check(test_sentence_2, block_size)
    if x == 1:    
        encoder = Encoder(vocab_size=tokenizer.vocab_size+1, d_model=d_model, 
                     sequence_length=sequence_length, num_heads=n_head, 
                     num_layers=n_layer, hidden_size=hidden_size).to(device)
        optimizer_cls = optim.Adam(encoder.parameters(), lr=learning_rate)
        decoder = Decoder(vocab_size=tokenizer.vocab_size+1, d_model=d_model, 
                     sequence_length=sequence_length, num_heads=n_head, 
                     num_layers=n_layer, hidden_size=hidden_size).to(device)
         # Initialize Utilities for attention visualization
        utils = Utilities(tokenizer, decoder)
        optimizer_lm = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                            lr=learning_rate)
        with open("speechesdataset/train_LM.txt", 'r', encoding='utf-8') as f:
            lm_train_text = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lm_train_text, block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        
        print("Starting training for language modeling task...")
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, attention_maps = decoder(xb, yb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            if i == 100 or i ==200 or i==300 or i==400:
                perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
                print(f"Iteration {i}/{max_iters},Perplexity: {perplexity:.4f}")
        perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
        print(f"Iteration {i}/{max_iters},Perplexity: {perplexity:.4f}")
       
        for ts in ['speechesdataset/test_LM_wbush.txt', 'speechesdataset/test_LM_obama.txt','speechesdataset/test_LM_hbush.txt']:
            with open(ts, 'r', encoding='utf-8') as f:
                ltt= f.read()
            
            test_loader = DataLoader(LanguageModelingDataset(tokenizer, ltt, block_size), batch_size=batch_size, shuffle=False)
            test_perplexity = compute_perplexity(decoder, test_loader, eval_iters=eval_iters)
            print(f"Perplexity on {ts}: {test_perplexity:.4f}")

        utilities = Utilities(tokenizer, decoder)
        sample_sentence = "China feels the winds of change."
        utilities.sanity_check(sample_sentence, block_size)

        
        num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"Number of parameters in the decoder: {num_params}")
       
        
        torch.save(decoder.state_dict(), 'decoder_model.pth')
        print("\nModels saved successfully!")
    if x == 2:    
        print("\nTraining Classification Task...")
        encoder = Encoder_3a(vocab_size=tokenizer.vocab_size+1, d_model=d_model, 
                     sequence_length=sequence_length, num_heads=n_head, 
                     num_layers=n_layer, hidden_size=hidden_size).to(device)
        optimizer_cls = optim.Adam(encoder.parameters(), lr=learning_rate)
        utils = Utilities(tokenizer, encoder) #remove
        for epoch in range(epochs_CLS):
            total_loss = 0
            for i, (xb, yb) in enumerate(train_CLS_loader):
                xb, yb = xb.to(device), yb.to(device)
                
                # Forward pass
                logits, loss, _ = encoder(xb, classify=True, targets=yb)
                
                # Backward pass
                optimizer_cls.zero_grad()
                loss.backward()
                optimizer_cls.step()
                
                total_loss += loss.item()
                
                if (i + 1) % eval_interval == 0:
                    avg_loss = total_loss / eval_interval
                    print(f'Classification Epoch [{epoch+1}/{epochs_CLS}], Step [{i+1}], Avg Loss: {avg_loss:.4f}')
                    total_loss = 0

            
            train_accuracy = compute_classifier_accuracy(encoder, train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(encoder, test_CLS_loader)

            print(f"Epoch {epoch+1}/{epochs_CLS}, Train classifier Accuracy: {train_accuracy:.2f}%, Test classifier Accuracy: {test_accuracy:.2f}%")
            # Sanity check
        utilities = Utilities(tokenizer,  encoder)
        test_sentence = "Let me tell you, you're not the only ones."
        utilities.sanity_check(test_sentence, block_size)
        # utilities = Utilities(tokenizer,  encoder)
        # test_sentence = "One issue Look at the world on this bright August night."
        # utilities.sanity_check(test_sentence, block_size)
        num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"Number of parameters in the encoder: {num_params}")
        torch.save(encoder.state_dict(), 'encoder_model.pth')
        print("\nModels saved successfully!")
        decoder = Decoder_3a(vocab_size=tokenizer.vocab_size+1, d_model=d_model, 
                     sequence_length=sequence_length, num_heads=n_head, 
                     num_layers=n_layer, hidden_size=hidden_size).to(device)
        utils = Utilities(tokenizer, decoder)
        optimizer_lm = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                            lr=learning_rate)
        with open("speechesdataset/train_LM.txt", 'r', encoding='utf-8') as f:
            lm_train_text = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lm_train_text, block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        
        print("Starting training for language modeling task...")
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, attention_maps = decoder(xb, yb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            if i == 100 or i ==200 or i==300 or i==400:
                perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
                print(f"Iteration {i}/{max_iters},Perplexity: {perplexity:.4f}")
        perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
        print(f"Iteration {i}/{max_iters},Perplexity: {perplexity:.4f}")
       
        for ts in ['speechesdataset/test_LM_wbush.txt', 'speechesdataset/test_LM_obama.txt','speechesdataset/test_LM_hbush.txt']:
            with open(ts, 'r', encoding='utf-8') as f:
                ltt= f.read()
            
            test_loader = DataLoader(LanguageModelingDataset(tokenizer, ltt, block_size), batch_size=batch_size, shuffle=False)
            test_perplexity = compute_perplexity(decoder, test_loader, eval_iters=eval_iters)
            print(f"Perplexity on {ts}: {test_perplexity:.4f}")

        utilities = Utilities(tokenizer, decoder)
        sample_sentence = "China feels the winds of change."
        utilities.sanity_check(sample_sentence, block_size)

        
        num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"Number of parameters in the decoder: {num_params}")
       
        
        torch.save(decoder.state_dict(), 'decoder_model_3a.pth')
        print("\nModels saved successfully!")
    if x == 3:
        print("\nTraining Classification Task...")
        encoder = Encoder_3b(vocab_size=tokenizer.vocab_size+1, d_model=d_model, 
                     sequence_length=sequence_length, num_heads=n_head, 
                     num_layers=n_layer, hidden_size=hidden_size).to(device)
        optimizer_cls = optim.Adam(encoder.parameters(), lr=learning_rate)
        utils = Utilities(tokenizer, encoder) #remove
        for epoch in range(epochs_CLS):
            total_loss = 0
            for i, (xb, yb) in enumerate(train_CLS_loader):
                xb, yb = xb.to(device), yb.to(device)
                
                # Forward pass
                logits, loss, _ = encoder(xb, classify=True, targets=yb)
                
                # Backward pass
                optimizer_cls.zero_grad()
                loss.backward()
                optimizer_cls.step()
                
                total_loss += loss.item()
                
                if (i + 1) % eval_interval == 0:
                    avg_loss = total_loss / eval_interval
                    print(f'Classification Epoch [{epoch+1}/{epochs_CLS}], Step [{i+1}], Avg Loss: {avg_loss:.4f}')
                    total_loss = 0

            
            train_accuracy = compute_classifier_accuracy(encoder, train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(encoder, test_CLS_loader)

            print(f"Epoch {epoch+1}/{epochs_CLS}, Train classifier Accuracy: {train_accuracy:.2f}%, Test classifier Accuracy: {test_accuracy:.2f}%")
            # Sanity check
        utilities = Utilities(tokenizer,  encoder)
        test_sentence = "Let me tell you, you're not the only ones."
        utilities.sanity_check(test_sentence, block_size)
        # utilities = Utilities(tokenizer,  encoder)
        # test_sentence = "One issue Look at the world on this bright August night."
        # utilities.sanity_check(test_sentence, block_size)
        num_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"Number of parameters in the encoder: {num_params}")
        torch.save(encoder.state_dict(), 'encoder_model.pth')
        print("\nModels saved successfully!")    
        decoder = Decoder_3b(vocab_size=tokenizer.vocab_size+1, d_model=d_model, 
                     sequence_length=sequence_length, num_heads=n_head, 
                     num_layers=n_layer, hidden_size=hidden_size).to(device)
        utils = Utilities(tokenizer, decoder)
        with open("speechesdataset/train_LM.txt", 'r', encoding='utf-8') as f:
            lm_train_text = f.read()
        train_LM_dataset = LanguageModelingDataset(tokenizer, lm_train_text, block_size)
        train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
        
        print("Starting training for language modeling task...")
        for i, (xb, yb) in enumerate(train_LM_loader):
            if i >= max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, attention_maps = decoder(xb, yb)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            optimizer.step()
            if i == 100 or i ==200 or i==300 or i==400:
                perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
                print(f"Iteration {i}/{max_iters},Perplexity: {perplexity:.4f}")
        perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters=eval_iters)
        print(f"Iteration {i}/{max_iters},Perplexity: {perplexity:.4f}")
       
        for ts in ['speechesdataset/test_LM_wbush.txt', 'speechesdataset/test_LM_obama.txt','speechesdataset/test_LM_hbush.txt']:
            with open(ts, 'r', encoding='utf-8') as f:
                ltt= f.read()
            
            test_loader = DataLoader(LanguageModelingDataset(tokenizer, ltt, block_size), batch_size=batch_size, shuffle=False)
            test_perplexity = compute_perplexity(decoder, test_loader, eval_iters=eval_iters)
            print(f"Perplexity on {ts}: {test_perplexity:.4f}")

        utilities = Utilities(tokenizer, decoder)
        sample_sentence = "China feels the winds of change."
        utilities.sanity_check(sample_sentence, block_size)

        
        num_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"Number of parameters in the decoder: {num_params}")
       
        
        torch.save(decoder.state_dict(), 'decoder_model_3b.pth')
        print("\nModels saved successfully!")

if __name__ == "__main__":
    main()