---
library_name: transformers
tags: []
---

# Model Card for FIMMJP

<!-- Provide a quick summary of what the model is/does. -->
FIMMJP is a neural recognition model for zero-shot inference of Markov jump processes (MJPs) on bounded state spaces from noisy and sparse observations.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
This model implements a neural recognition model for zero-shot inference of MJPs, which are continuous-time stochastic processes describing dynamical systems evolving in discrete state spaces. The model is based on the paper "Foundation Inference Models for Markov Jump Processes" (https://arxiv.org/abs/2406.06419).

- **Model type:** Neural recognition model for MJPs

### Model Sources [optional]

<!-- Provide the basic links for the model. -->
- **Repository:** https://github.com/cvejoski/OpenFIM
- **Paper:** https://arxiv.org/abs/2406.06419

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->
The model can be used directly for zero-shot inference of MJPs in various applications such as molecular simulations, experimental ion channel data, and simple protein folding models.

## How to Get Started with the Model
1. Installing the library

```bash
# Clone the repository
git clone https://github.com/cvejoski/OpenFIM.git

# Navigate to the project directory
cd OpenFIM

# Install the required dependencies
pip install -e .

```

2. Using the model

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the pre-trained model and tokenizer
model_name = "cvejoski/FIMMJP"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example input
input_text = "Your input text here"
inputs = tokenizer(input_text, return_tensors="pt")

# Perform inference
outputs = model(**inputs)
logits = outputs.logits

# Process the output as needed
print(logits)
```

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

[More Information Needed]

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

[More Information Needed]

### Results

[More Information Needed]

#### Summary


## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]