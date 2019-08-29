import langid
import spacy
import en_core_web_sm ## spacy's english model 
import pickle 
import pandas as pd
from spacy.matcher import PhraseMatcher 
from spacy.lang.en import English
from spacy.tokens import Span,Doc
from translate import Translator

### open dataset file (using with statement which is cleaner for exception handling rather than try and finally also ensure proper acquisition and release of resources
with open("dataset/mypersonality_final.csv",encoding='latin') as f:
        DATA=f.read()

### sample text
# texts = ["halo nama saya Reinhard. Saat ini saya sedang membaca buku Ready Player One sambil mendengarkan album How to Pimp a Butterfly oleh Kendrick Lamar"]
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

### load the installed model "en_core_web_sm"

# nlp = spacy.load('en_core_web_sm')

# docs = list(nlp.pipe(texts))

# for token in doc:
#         print(type(token),token.lemma_.strip())

# print('without punctuations: ',[iter(doc) for doc in docs])

# print('without punctuations: ',[doc.Token.lemma_ for doc in docs if not doc.Token.is_punct])

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

# Load dataset with pandas 
df = pd.read_csv('dataset/mypersonality_final.csv',delimiter=',',encoding='latin',header=0)
personality = df.iloc[0:,list(range(7,12))]
ext = personality['cEXT'].apply(lambda x:x.replace('y','1') if x=='y' else x.replace('n','0'))
neu = personality['cNEU'].apply(lambda x:x.replace('y','1') if x=='y' else x.replace('n','0'))
agr = personality['cAGR'].apply(lambda x:x.replace('y','1') if x=='y' else x.replace('n','0'))
con = personality['cCON'].apply(lambda x:x.replace('y','1') if x=='y' else x.replace('n','0'))
opn = personality['cOPN'].apply(lambda x:x.replace('y','1') if x=='y' else x.replace('n','0'))
personality = pd.concat([ext,neu,agr,con,opn],ignore_index=True,axis=1) ### gotta concat 'em all !, axis 1 == treating copy vertically(column wise), 
                                                                        ### ignore index == ignore index from original source
columnNames=['bEXT','bNEU','bAGR','bCON','bOPN']
personality.columns = columnNames
personality[columnNames] = personality[columnNames].astype(int)
df = pd.concat([df,personality],axis=1)

print(df.head())

STATUS = df['STATUS']
AUTHID = [{"AUTHID":el} for el in df['#AUTHID']]

FEED = list(zip(list(STATUS),AUTHID))

#Instantiate spacy model 
# nlp = English()
nlp = spacy.load('en_core_web_sm')

#matcher using shared vocab 
matcher = PhraseMatcher(nlp.vocab)
matcher.add("PROPER_NAME",None,nlp.make_doc("*PROPNAME*"))

#custom pipeline 
def propname_component(doc):
        matches = matcher(doc)
        spans = [Span(doc,start,end,label="PROPER_NAME")for match_id,start,end in matches]
        doc.ents = list(doc.ents) + spans
        return doc

#add it to the existing pipeline
nlp.add_pipe(propname_component,after='ner')

#disable dependency parser and pos tagger
nlp.disable_pipes("tagger","parser")
print(nlp.pipe_names)

Doc.set_extension('AUTHID',default=None)

buff = []
for doc, context in nlp.pipe(FEED,as_tuples=True):
        doc._.AUTHID = context['AUTHID']
        buff.append(doc)
        #buff.append(''.join(['{0}{1}'.format(token.text,token.whitespace_) for token in doc if not token.is_stop]))

print(buff)
print(len(buff))

with open('dataset/doc-with-authid.p','wb') as f:
    pickle.dump(buff,f)

print("pickled in dataset")

#with open('dataset/token_stopwords.p','wb') as f:
#    pickle.dump(buff,f)
#
#print("pickled in dataset")
