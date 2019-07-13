import langid
import spacy
from translate import Translator

text = "halo nama saya Reinhard. Saat ini saya sedang membaca buku Ready Player One sambil mendengarkan album How to Pimp a Butterfly oleh Kendrick Lamar"
parameter = {
        'from_lang':langid.classify(text)[0],
        'to_lang':'en'
        }
translator = Translator(**parameter)
textTranslated = translator.translate(text)
print(textTranslated)

