import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud,STOPWORDS

### load datest from csv to pandas.dataframe
df = pd.read_csv('./dataset/mypersonality_final.csv',encoding='latin-1',header=0)

### filter the df columns and change it to binary
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

### print df 5 top rows sample
print(df.head())

### describe columns with numeric values from df 
print(df.describe())

### sum all personality & sort it
personalitySum = personality.iloc[:].sum().sort_values()

#personalityNew =[personality.loc[personality[x]==1].sum() for x in columnNames]
#print(personalityNew)
### save it for later !

### for all unique personality supposedly 2^5=32 unique personalities but theres only 31 in this dataset
buff = set([tuple(personality.iloc[x].values) for x in range(0,personality.shape[0])]) ### return unique set from the personality dataframe values, casted to tuple bacause iloc return numpy.ndarray its unhashable 
popBuff = []
test = []
i = 0
while len(buff) > 0:
    pop = buff.pop()
    test.append(personality.loc[(personality[columnNames[0]]==pop[0])&(personality[columnNames[1]]==pop[1])&(personality[columnNames[2]]==pop[2])&(personality[columnNames[3]]==pop[3])&(personality[columnNames[4]]==pop[4])])
    popBuff.append([pop,test[i].count().values[0]])
    i+=1
popBuff.sort(key=lambda x: x[1]) ### sort it!

### check who got 00000
check = df.loc[(df[columnNames[0]]==0)&(df[columnNames[1]]==0)&(df[columnNames[2]]==0)&(df[columnNames[3]]==0)&(df[columnNames[4]]==0)]
print(check.describe())

### check whos got 11111
checkN = df.loc[(df[columnNames[0]]!=0)&(df[columnNames[1]]!=0)&(df[columnNames[2]]!=0)&(df[columnNames[3]]!=0)&(df[columnNames[4]]!=0)]
print(checkN.describe())

### check anyone elses
ic = check.index.values.tolist()
idf = df.index.values.tolist()
[idf.remove(x) for x in ic]
icN = checkN.index.values
checkA = df.iloc[idf]
print(checkA.describe())

#x = zip(popBuff[0][0],columnNames)
#print(set(x))
#print(popBuff[0][0])
#print(personality.shape[0])
#print(buff.pop()[0])

### settings for plot of unique personalities
categories = ['({0})'.format(''.join(map(str,x[0]))) for x in popBuff]
popBuffCount = [popBuff[x][1] for x in range(31)]
sns.set(font_scale=0.5)  
plt.figure(figsize=(15,8))
ax = sns.barplot(categories,popBuffCount)
plt.title("statuses in each 31 unique personalities",fontsize=24)
plt.ylabel("number of statuses",fontsize=18)
plt.xlabel("personality",fontsize=18)
rects = ax.patches ### return rectangular objects (matplotlib.patches.Rectangle)
labels = popBuffCount  
for rect,label in zip(rects,labels):
    height = rect.get_height()### return height with the y-axis's magnitude
    ### add text to each objects in the plot
    ax.text(rect.get_x() + rect.get_width()/2, height+5,label,ha='center',va='bottom',fontsize=10)

### settings for plot of personalities
categories = personalitySum.index.values
sns.set(font_scale=2)
plt.figure(figsize=(8,6))
ax = sns.barplot(categories,personalitySum.values)
plt.title("statuses in each personality",fontsize=24)
plt.ylabel("number of statuses",fontsize=18)
plt.xlabel("personality",fontsize=18)
rects = ax.patches ### return rectangular objects (matplotlib.patches.Rectangle)
labels = personality.iloc[:].sum().values
for rect,label in zip(rects,labels):
    height = rect.get_height()### return height with the y-axis's magnitude
    ### add text to each objects in the plot
    ax.text(rect.get_x() + rect.get_width()/2, height+5,label,ha='center',va='bottom',fontsize=18)

###unused
#pm = pd.melt(personality)
#print(pm)
#ax = sns.countplot(data=pm.loc[pm['value']==1],x='variable')
STOP_WORDS.add("PROPNAME")
### settings for wordcloud 
plt.figure(figsize=(40,25))
subsetOpn = df[df.cOPN=='y']
textOpn = subsetOpn.STATUS.values
cloud_OPN = WordCloud(
        stopwords=STOP_WORDS,
        background_color='black',
        collocations=False,
        width=2500,
        height=1800
        ).generate(" ".join(textOpn))
plt.axis('off')
plt.title("OPENNESS",fontsize=40)
plt.imshow(cloud_OPN)

### settings for wordcloud 
plt.figure(figsize=(40,25))
subset = df[df.cCON=='y']
text = subset.STATUS.values
cloud_CON = WordCloud(
        stopwords=STOP_WORDS,
        background_color='black',
        collocations=False,
        width=2500,
        height=1800
        ).generate(" ".join(text))
plt.axis('off')
plt.title("CONSCIENTIOUSNESS",fontsize=40)
plt.imshow(cloud_CON)

### settings for wordcloud 
plt.figure(figsize=(40,25))
subset = df[df.cEXT=='y']
text = subset.STATUS.values
cloud_EXT = WordCloud(
        stopwords=STOP_WORDS,
        background_color='black',
        collocations=False,
        width=2500,
        height=1800
        ).generate(" ".join(text))
plt.axis('off')
plt.title("EXTRAVERSION",fontsize=40)
plt.imshow(cloud_EXT)

### settings for wordcloud 
plt.figure(figsize=(40,25))
subset = df[df.cAGR=='y']
text = subset.STATUS.values
cloud_AGR = WordCloud(
        stopwords=STOP_WORDS,
        background_color='black',
        collocations=False,
        width=2500,
        height=1800
        ).generate(" ".join(text))
plt.axis('off')
plt.title("AGREEABLENESS",fontsize=40)
plt.imshow(cloud_AGR)

### settings for wordcloud 
plt.figure(figsize=(40,25))
subset = df[df.cNEU=='y']
text = subset.STATUS.values
cloud_NEU = WordCloud(
        stopwords=STOP_WORDS,
        background_color='black',
        collocations=False,
        width=2500,
        height=1800
        ).generate(" ".join(text))
plt.axis('off')
plt.title("NEUROTICISM",fontsize=40)
plt.imshow(cloud_NEU)

plt.show()
