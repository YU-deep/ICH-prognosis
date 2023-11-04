"""" if you want to run BioClinical BERT separately,
you need external run .sh file and configure corresponding environment
you can find relevant information in
www.github.com/EmilyAlsentzer/clinicalBERT
"""

from transformers import AutoTokenizer, AutoModel


def get_Text_Encoder():
    """Here, we use pre-trained BioClinical BERT"""
    encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return encoder