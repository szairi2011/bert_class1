# Importing necessary libraries
import streamlit as st  # Streamlit for creating a web-based application
from transformers import BertTokenizer, BertForSequenceClassification  # Hugging Face transformers library
import torch  # PyTorch for deep learning operations
import re  # Regular expressions for pattern matching and text manipulation

# Cache the BERT tokenizer and the pre-trained model for sequence classification
@st.cache(allow_output_mutation=True)
def get_model():
    # Initialize the tokenizer for BERT
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # Load a pre-trained Bert model from a specific path 
    model = BertForSequenceClassification.from_pretrained("hams2/bertclass")
    return tokenizer, model  # Return the tokenizer and the model

# Retrieve the cached tokenizer and model
tokenizer, model = get_model()

# Set the title for the Streamlit application
st.title('Sequence Classification')  

# Text area to get the SGW operation input from the user
sgw_op = st.text_area('SGW Operation')  
# Button to trigger the prediction process
button = st.button("Predict")

# If the SGW operation has input and the button is clicked, proceed with prediction
if sgw_op and button:
    # Clean the input by removing newline characters
    sgw_op1 = sgw_op.replace('\n', '')
    # Remove certain special characters from the sequence using regex
    pattern = r'[<\"/]'
    cleaned_seq1 = re.sub(pattern, '', sgw_op1)
    # Replace '>' with a space 
    cleaned_seq = cleaned_seq1.replace('>', ' ')
    
    # Put the model into evaluation mode
    model.eval()
    
    # Tokenize the cleaned sequence, setting padding, maximum length, and truncation
    encoding = tokenizer(cleaned_seq, return_tensors='pt', max_length=512, padding='max_length', truncation=True)
    
    # Get input IDs and attention mask from the tokenized output
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    
    # Perform evaluation with the model to get prediction logits
    with torch.no_grad():  # Disable gradient computation for evaluation
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # Output logits from the model
        
        # Get the index of the highest logit value (predicted class)
        _, preds = torch.max(logits, dim=1)
    
    # Display the result and provide appropriate links based on the predicted class
    if preds.item() == 0:  # If the predicted class is 0
        st.write("Workflow: Full Service")  # Workflow classification
        # Link to the specific prediction app
        st.write("Link to predict CCP Message: [https://fullservice-eys93zac7jrtbgdw9fs4w7.streamlit.app/](https://fullservice-eys93zac7jrtbgdw9fs4w7.streamlit.app/)")
    
    elif preds.item() == 1:  # If the predicted class is 1
        st.write("Workflow: Mark As Give Up")  # Workflow classification
        # Link to the prediction app
        st.write("Link to predict CCP Messages: [https://markasgup-mymgunzherqqedw3hkf5wq.streamlit.app/](https://markasgup-mymgunzherqqedw3hkf5wq.streamlit.app/)")
    
    elif preds.item() == 2:  # If the predicted class is 2
        st.write("Workflow: Give Up")  # Workflow classification
        # Link to the prediction app
        st.write("Link to predict CCP Message: [https://giveup-jn8r7oaepixsnulwzmwffu.streamlit.app/](https://giveup-jn8r7oaepixsnulwzmwffu.streamlit.app/)")
    
    elif preds.item() == 3:  # If the predicted class is 3
        st.write("Workflow: Split On Two Accounts")  # Workflow classification
        # Link to the prediction app
        st.write("Link to predict CCP Messages: [https://split6cv6zqcuioigt8xp.streamlit.app/](https://split6cv6zqcuioigt8xp.streamlit.app/)")
    
    elif preds.item() == 4:  # If the predicted class is 4
        st.write("Workflow: Split On Three Accounts")  # Workflow classification
        # Link to the prediction app
        st.write("Link to predict CCP Messages: [https://split6cv6zqcuioigt8xp.streamlit.app/](https://split6cv6zqcuioigt8xp.streamlit.app/)")
    
    elif preds.item() == 5:  # If the predicted class is 5
        st.write("Workflow: Split On Four Accounts")  # Workflow classification
        # Link to the prediction app
        st.write("Link to predict CCP Message: [https://split6cv6zqcuioigt8xp.streamlit.app/](https://split6cv6zqcuioigt8xp.streamlit.app/)")