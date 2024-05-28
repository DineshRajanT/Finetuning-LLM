**Phi-2 Model Fine-Tuning for Question Answering**

**Overview:**

This notebook demonstrates the process of fine-tuning the Microsoft Phi-2 model for a Question Answering (QA) task using the Hugging Face library. The Phi-2 model is loaded in a 4-bit quantized format to optimize memory usage, and the training is enhanced using the Low-Rank Adaptation (LORA) technique. We also cover the setup, dataset preparation, and training of the model, and finally, we perform inference to test the fine-tuned model.

**Table of Contents**
    * Setup and Installation
    * Model Loading
    * Hugging Face Login
    * Dataset Preparation
    * Tokenization
    * Training Setup
    * Model Training
    * Inference

**1. Setup and Installation**
Explanation:
Necessary libraries are installed, including einops, datasets, bitsandbytes, accelerate, peft, and flash_attn.
The transformers library is reinstalled from the Hugging Face GitHub repository to ensure the latest version is used.
Torch is upgraded to the latest version.
gc.collect() is used to clean up and free memory.
!nvidia-smi checks the GPU details.


**2. Model Loading**
We load the Microsoft Phi-2 model from Hugging Face in a 4-bit quantized format and set up the tokenizer.
Explanation:
BitsAndBytesConfig is configured to load the model in 4-bit quantized format to reduce memory usage.
AutoModelForCausalLM loads the Phi-2 model.
AutoTokenizer sets up the tokenizer with appropriate configurations like adding an EOS token and padding token.


**3. Hugging Face Login**
We log in to Hugging Face to save the updated model weights after training.
Explanation:
notebook_login() prompts the user to log in to Hugging Face using an access token with write permissions. This step is necessary to save the fine-tuned model to the Hugging Face Hub.


**4. Dataset Preparation**
We load a slice of the WebGLM dataset for training and merge the validation/test datasets.
python. load_dataset loads the WebGLM QA dataset from Hugging Face.
We use a specific slice of the training dataset (train[5000:10000]) and the entire test dataset.


**5. Tokenization**
We define a function to create prompts from the dataset and tokenize them.
Explanation:
collate_and_tokenize function creates a prompt using the question, answer, and references from the dataset and tokenizes it.
Tokenized datasets are created for both training and testing by mapping the collate_and_tokenize function.


**6. Training Setup**
We set up the model for training using LORA and prepare for gradient checkpointing to save memory.
Explanation:
FullyShardedDataParallelPlugin and Accelerator are set up to enable efficient training.
print_trainable_parameters function prints the number of trainable parameters in the model.
Gradient checkpointing is enabled to save memory.
The model is prepared for k-bit training using prepare_model_for_kbit_training.
LORA configuration is set, and the model is wrapped with LORA using get_peft_model.


**7. Model Training**
We define training arguments and initiate the training process.
Explanation:
TrainingArguments are defined for the training process.
Trainer class is used to handle the training loop.
The training time is calculated and printed.
The fine-tuned model is saved to the Hugging Face Hub.


**8. Inference**
We perform inference using the trained model.
Explanation:
A new prompt is defined for testing.
The prompt is tokenized and fed into the model for inference.
The model generates a response which is then decoded and printed.
The fine-tuned model is loaded from the Hugging Face Hub and used for inference.

