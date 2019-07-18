import langid
import spacy
import en_core_web_sm ## spacy's english model 
from translate import Translator

### sample text
texts = ["halo nama saya Reinhard. Saat ini saya sedang membaca buku Ready Player One sambil mendengarkan album How to Pimp a Butterfly oleh Kendrick Lamar"]

def coroutine(func):
    def start(*args, **kwargs):
        cr = func(*args, **kwargs)
        next(cr)
        return cr
    return start

def source(texts,targets):
        for text in texts:
                for t in targets:
                        t.send(text)

# #### parameters for Translator (do it in production,because dev using dataset which already in english)
# parameter = {
#         'from_lang':langid.classify(texts[0])[0],
#         'to_lang':'en'
#         }

# ### load the installed model "en_core_web_sm"
# nlp = spacy.load('en_core_web_sm')

# ### check spacy's nlp pipeline :)
# print(nlp.pipe_names)

# ### instantiate Translator with specified parameters (** -> see upacking in python)
# translator = Translator(**parameter)

# ### static method return translated text 
# #text = translator.translate(text)

# # def my_translate(doc):
#         # print(type(doc))
# # 
#         # return doc

# #### add pipe 
# # nlp.add_pipe(my_translate, first = True)

# ### text processed with spacy nlp pipeline 
# #doc = nlp(text)

# @coroutine
# def my_translate(targets):
#         text = "placeholder"
#         while True:
#                 text = yield text
#                 lang = langid.classify(text)[0]
#                 translator = Translator(from_lang=lang,to_lang='en')
#                 text = translator.translate(text)
#                 for target in targets:
#                         target.send(text)
# @coroutine
# def printer():
#         while True:
#                 line = (yield)
#                 print (line)

# source(texts,targets=[my_translate(targets=[
#         printer()])
# ])

# x = my_translate([])
# print(x.send(texts[0]))
