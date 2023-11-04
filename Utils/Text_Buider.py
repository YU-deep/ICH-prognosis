import re
from sklearn import preprocessing
import torch
from transformers import BertTokenizer
from transformers import AutoTokenizer, AutoModel

def get_patient_text(file_path):
    text = ''
    file_name = re.sub('\s+', ' ', file_path)
    name_piece = file_name.split(' ')
    age = name_piece[2][1:]
    text += 'Age:'+age + '\n'
    if name_piece[2][0] == 'M':
        text += 'Gender:Man'
    else:
        text += 'Gender:Female'

    time = name_piece[5]
    if time.endswith('小时'):
        time_min = str(int(60 * float(time.split('小时')[0])))
    elif "小时" not in time:
        time_min = str(int(float(time.split('分钟')[0])))
    else:
        time_temp = time.split('小时')
        time_min = str(int(60 * float(time_temp[0]) + float(time_temp[1].split('分钟')[0])))
    text += 'Onset-To-CT:' + time_min + 'min\n'

    stay = name_piece[7].replace('入院时间', '').replace('住院时间', '').replace('住院天数', '').replace('天', '')
    text += 'Hospital Stay:' + stay + 'days\n'

    GCS = name_piece[6].replace('分', '').replace('GCS', '')
    text += 'GCS Score:'+ GCS +'\n'

    treatment_method = name_piece[8]
    text += 'Treatment Method:' + treatment_method

    text += 'Doctor\'s Diagnosis:' + name_piece[-1]
    return text


# def text_to_tensor(text):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
#     bert_input = tokenizer(text, padding='max_length',
#                            max_length=10,
#                            truncation=True,
#                            return_tensors="pt")
#     return bert_input

def text_to_tensor(text):
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    text_tensor = tokenizer(text, padding='max_length',
              max_length=10,
              truncation=True,
              return_tensors="pt")
    return text_tensor