import pandas as pd
import torch
from transformers import MobileBertTokenizer, MobileBertModel
import string
from datasets import Dataset as Hug_Face_Dataset,load_from_disk
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
import torch.nn as nn      
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
import os 
import argparse
# imported the libraries needed

def dataProcessing(dataset,tokenizer,max_tokens,batch_size,mapping): # function for generating dataloader for both train and test
    inputData = dataset['input'] 
    outputData = dataset['output']
    myDataset = MyDataset(inputData,outputData,tokenizer,max_tokens,mapping)
    dataLoader = DataLoader(myDataset,batch_size,shuffle = False)
    return dataLoader    

class MyDataset(Dataset): # Class for preparing mobileBERT tokens from datasets and load the data batchwise

    def __init__(self, inputTokens,outputEntity,tokenizer,max_tokens=128,mapping={'B-PER':1}):
        self.source = inputTokens
        self.target = outputEntity
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.mapping = mapping
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def __len__(self):
        return len(self.source)
    
    def __getitem__(self, idx):
        
        source_data = self.source[idx]
        target_data = self.target[idx]
        tokenized_input = self.tokenizer(source_data, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_tokens,is_split_into_words=True)
        label_ids = torch.zeros(tokenized_input['input_ids'].size(1), dtype=torch.long)
        
        for start, end, label in target_data:
            start = int(start)
            end = int(end)
            label_id = self.mapping[label]
            label_ids[start:end+1] = label_id
        
        return tokenized_input['input_ids'].squeeze(0).to(self.device), tokenized_input['attention_mask'].squeeze(0).to(self.device), label_ids.to(self.device)

class NERModel(nn.Module): #Model class for architecture with 3 bidirectional GRU's and 1 fully connected component.
    
    def __init__(self,hidden_size=64, num_entities=41,mobilebert_model=None,dropout_p=0.5):
        
        super(NERModel, self).__init__()
        
        self.mobilebert_model = mobilebert_model
        
        # GRU
        self.gru = nn.GRU(input_size=self.mobilebert_model.config.hidden_size, hidden_size=hidden_size,num_layers = 3, bidirectional=True, batch_first=True)
        
        
        # Fully connected layers
        self.fc_layer = nn.Linear(2*hidden_size, num_entities)
        
        self.dropout = nn.Dropout(dropout_p) #droppout to decrease overfitting 
    
            
    def forward(self, input_ids, attention_mask):
        
        batch_size = input_ids.size(0)
        
        outputs = self.mobilebert_model(input_ids=input_ids, attention_mask=attention_mask)
        
        sequence_output = outputs.last_hidden_state 
        
        gru_output,_ = self.gru(sequence_output) 
        
        output = self.dropout(gru_output)
        
        overallOutput = self.fc_layer(output)
        
        return overallOutput

def testingAccuracyCalculator(model,tokenizer,max_tokens,batch_size,test_dataLoader,test_numberOfNonZeroLabelledInputs,test_input):
    #function used to calculate and log test data accuracies, losses and performance metrics.
    
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    
    correct_predictions = 0.0        
    all_true_labels = []
    all_pred_labels = []
    
    model.eval()
    
    with torch.no_grad():
        
        for idx,(input_ids, attention_mask, labels) in enumerate(tqdm(test_dataLoader)):

            output = model(input_ids, attention_mask)

            # Flatten the logits and labels for loss calculation

            output_flat = output.view(-1, output.size(-1))
            labels_flat = labels.view(-1)

            probabilities = F.softmax(output_flat, dim=1)

            predicted_labels = torch.argmax(probabilities, dim=1)

            correct_predictions += ((predicted_labels == labels_flat) & (labels_flat!=0)).sum().item() #counting only actual(useful) tokens
            
            non_zero_mask = (labels_flat != 0)
            
            filtered_labels = labels_flat[non_zero_mask] #filtering non zero labels which corresponds to actual tokens.
            
            filtered_predictions = predicted_labels[non_zero_mask] #filtering predicted labels corresponding to actual tokens.

            all_true_labels.extend(filtered_labels.cpu().numpy())
            all_pred_labels.extend(filtered_predictions.cpu().numpy())

            loss = criterion(output_flat, labels_flat) #corss entropy loss calculation

            total_loss += loss.item() 
        
    avg_loss = total_loss / test_input
    accuracy_percentage = (correct_predictions / test_numberOfNonZeroLabelledInputs)*100
    precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_pred_labels, average='weighted',zero_division=0) #I have used weighted average of three scores.
    print(f'average_loss {avg_loss}, accuracyPercentage, {accuracy_percentage:.2f}')  
    print(f'precision{precision:.2f},recall{recall:.2f},f1_score{f1:.2f}')
    with open(r"/content/logs", 'a') as log_file:  # change the file path as required
        log_file.write(f'Testing mode ->> avg_loss {avg_loss}, accuracy_percentage {accuracy_percentage:.2f},precision {precision:.2f},recall {recall:.2f}, f1 {f1:.2f}\n')
    model.train()

def train(max_tokens,hidden_size,learning_rate,batch_size,epochs,dropout): 

    num_entities = 41 # There are 41 distinct entities in overall multilingual datasets including label -> 0 for stop words, punctuaitons and padded tokens
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased') #loaded pretrained mobileBERT_tokenizer 
    mobilebert_model = MobileBertModel.from_pretrained('google/mobilebert-uncased') ##loaded pretrained mobileBERT_model
    
    train_dataset = load_from_disk(r"/content/train") #loaded train dataset from hugging face object

    mapping = {'B-PER':1,'I-PER':2,'B-ORG':3,'I-ORG':4,'B-LOC':5,'I-LOC':6,'B-MISC':7,'I-MISC':8,'B-NRM':9,'B-REG':10,'B-RS':11,'I-LIT':12,'I-NRM':13,'I-REG':14,'I-RS':15,'B-ANIM':16,'I-ANIM':17,'B-BIO':18,'I-BIO':19,'B-CEL':20,'I-CEL':21,'B-DIS':22,'I-DIS':23,'B-EVE':24,
      'I-EVE':25,'B-FOOD':26,'I-FOOD':27,'B-INST':28,'I-INST':29,'B-MEDIA':30,'I-MEDIA':31,'B-MYTH':32,'I-MYTH':33,'B-PLANT':34,'I-PLANT':35,'B-TIME':36,'I-TIME':37,'B-VEHI':38,'I-VEHI':39,'B-LIT':40}
    
    train_dataloader = dataProcessing(train_dataset,tokenizer,max_tokens,batch_size,mapping)
    
    test_dataset = load_from_disk(r"/content/test") #loaded test dataset form hugging face object
    
    test_dataLoader = dataProcessing(test_dataset,tokenizer,max_tokens,batch_size,mapping)
    
    train_input = len(train_dataset['input'])

    test_input = len(test_dataset['input'])

    
    train_numberOfNonZeroLabelledInputs = 0
    for temp in train_dataset['output']:
        train_numberOfNonZeroLabelledInputs+=len(temp)

    test_numberOfNonZeroLabelledInputs = 0
    for temp in test_dataset['output']:
        test_numberOfNonZeroLabelledInputs+=len(temp)

    model = NERModel(hidden_size,num_entities,mobilebert_model,dropout)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    criterion = nn.CrossEntropyLoss()
    
    model.to(device) #if GPU is available model will be trained on GPU for faster computation.
    
    model.train()
    
    with open(r"/content/logs/logs.log", 'w') as log_file: #creating log file
        log_file.write("Starting logs\n")
    
    
    for epoch in range(epochs): 

        total_loss = 0.0
        
        correct_predictions = 0.0
        print(f'epoch{epoch}')
        
        all_true_labels = []
        all_pred_labels = []
        
        for idx,(input_ids, attention_mask, labels) in enumerate(tqdm(train_dataloader)):
            
            optimizer.zero_grad()
        
            output = model(input_ids, attention_mask)

            # Flatten the logits and labels for loss calculation
            
            output_flat = output.view(-1, output.size(-1))
            labels_flat = labels.view(-1)
        
            probabilities = F.softmax(output_flat, dim=1)
        
            predicted_labels = torch.argmax(probabilities, dim=1)

            correct_predictions += ((predicted_labels == labels_flat) & (labels_flat!=0) ).sum().item()
            
            non_zero_mask = (labels_flat != 0)
            
            filtered_labels = labels_flat[non_zero_mask]
            
            filtered_predictions = predicted_labels[non_zero_mask]

            all_true_labels.extend(filtered_labels.cpu().numpy())
            all_pred_labels.extend(filtered_predictions.cpu().numpy())

            # Calculate loss
            loss = criterion(output_flat, labels_flat)
            
            if torch.isnan(loss): # check for loss = nan
                continue
            # Backward pass
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5) # gradient clipping
            
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item() 
            
            if(idx%200==0):
                print(f'loss {loss.item()}') # Printing loss to check intermediate results.

        # Print average loss
        avg_loss = total_loss / train_input
        accuracy_percentage = (correct_predictions / train_numberOfNonZeroLabelledInputs)*100
        precision, recall, f1, _ = precision_recall_fscore_support(all_true_labels, all_pred_labels, average='weighted',zero_division=0) #I have used weighted average of three scores.
        print(f'epoch {epoch}, average_loss {avg_loss}, accuracyPercentage, {accuracy_percentage:.2f}')  
        print(f'precision{precision:.2f},recall{recall:.2f},f1_score{f1:.2f}')
        with open(r"/content/logs/logs.log", 'a') as log_file:
            log_file.write(f'Training mode ->> epoch {epoch},avg_loss {avg_loss}, accuracy_percentage {accuracy_percentage:.2f},precision {precision:.2f},recall {recall:.2f}, f1 {f1:.2f}\n')
        checkpoint_path = os.path.join(r"/content/logs", f'epoch_{epoch}_checkpoint.pth')
        torch.save(model.state_dict(), checkpoint_path) # Saving model for each epoch
        testingAccuracyCalculator(model,tokenizer,max_tokens,batch_size,test_dataLoader,test_numberOfNonZeroLabelledInputs,test_input)

def parse_arguments():

    args = argparse.ArgumentParser(description='Training Parameters')
    
    args.add_argument('-mt', '--maxTokens', type= int, default=128,help='Choice of maximum number of tokens')
    
    args.add_argument('-hs', '--hiddenSize', type= int, default=64,help='Choice of hidden size')

    args.add_argument('-lr', '--learningRate', type= float, default=1e-4, help='Choice of learing rate')

    args.add_argument('-bs', '--batchSize', type= int, default=64, choices = [32,64,128,256,512,1024],help='Choice of batch size')
    
    args.add_argument('-ep', '--epochs', type= int, default=10,help='Choice of epochs')

    args.add_argument('-dr', '--dropout', type= float, default=0.5, help='Choice of dropout')

    return args.parse_args()

args = parse_arguments()

train(args.maxTokens,args.hiddenSize,args.learningRate,args.batchSize,args.epochs,args.dropout) 
