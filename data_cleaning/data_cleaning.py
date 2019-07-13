import langid
import spacy
import en_core_web_sm ## spacy's english model 
from translate import Translator

### sample text
text = "halo nama saya Reinhard. Saat ini saya sedang membaca buku Ready Player One sambil mendengarkan album How to Pimp a Butterfly oleh Kendrick Lamar"

#### parameters for Translator
parameter = {
        'from_lang':langid.classify(text)[0],
        'to_lang':'en'
        }
### load the installed model "en_core_web_sm"
nlp = spacy.load('en_core_web_sm')

### check spacy's nlp pipeline :)
print(nlp.pipe_names)

### instantiate Translator with specified parameters (** -> see upacking in python)
translator = Translator(**parameter)

### static method return translated text 
textTranslated = translator.translate(text)

### text processed with spacy nlp pipeline 
doc = nlp(textTranslated)

print(textTranslated)
print(doc[0])
