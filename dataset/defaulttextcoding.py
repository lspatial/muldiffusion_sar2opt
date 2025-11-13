from pixels2text import TextInfoRetriever
import torch, open_clip
from PIL import Image
from datasampling_txt import pretrainedModel, retrieveTextCode 
import numpy as np  


def defaulttextcode(promt = 'A remote sensing optical image'):
    tokenizer,remoteclip,_ = pretrainedModel()
    textfeatures0 = retrieveTextCode(tokenizer,remoteclip,promt)
    textfeatures0 = list(textfeatures0)
    textfeatures = np.float32(textfeatures0)
    return textfeatures 
  
def generateDefaultTextCode(prompt='A remote sensing optical image',
    outfl='/devb/sar2opt_diff_txt/test/deftextcode.npy'):
    textfeature = defaulttextcode(prompt)
    if outfl is None:
        outfl = 'default_text_code.npy'
    np.save(outfl, textfeature)
    

if __name__=='__main__':
    generateDefaultTextCode() 

