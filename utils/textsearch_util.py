import unicodedata
from Levenshtein import distance

special_chars = ['!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/',':',';','<','=','>','?','@','[','\\',']','^','_','`','{','|','}','~']


def strip_accents(s):
   s = str(s)
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def clean_str(str, noise=[]):
    for c in special_chars:
        str = str.replace(c, "")
    str = [tx.strip().lower() for tx in str.split()]
    for word in noise:
        if word in str:
            str.remove(word)
    return str

def noisy_words(place_list):
    place_vocab = []
    noise = []
    for place in place_list.values():
        place_vocab.extend([word.strip().lower() for word in place.split()])
    for word in place_vocab:
        if word[-1] == 's':
            count = place_vocab.count(word) + place_vocab.count(word[:-1])
        else:
            count = place_vocab.count(word) + place_vocab.count(word + 's')

        if count > 3 and word not in noise:
            noise.append(word)
            if place_vocab.count(word + 's') >= 1:
                noise.append(word+'s')
    return noise

def get_score(str1, str2):
    dist = distance(str1.lower(),str2.lower())
    #score = 1 if dist == 0 else (len(str1)-dist)/len(str1) 
    score = 1 - (dist/max(len(str1),len(str2))) 
    return score

def calculate_sim_score(text, pname, noise=[]):
    text = clean_str(text)
    pname = clean_str(pname, noise=noise)
    word_score = 0
    for i in range(len(text)):
        for j in range(len(pname)):
            for x in range(j+1):
                score = get_score(''.join(text[:i+1]), ''.join(pname[x:j+1]))
                word_score = max(word_score,score)
    return round(word_score,3)