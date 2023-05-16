from random import choice, randint
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from functions import *

lemmatizer = WordNetLemmatizer()

nouns_list, nouns_set = load_vocab('../img2text/nouns.txt')
adjs_list, adjs_set = load_vocab('../img2text/adjectives.txt')
verbs_list, verbs_set = load_vocab('../img2text/verbs.txt')


def gen_a_new_prompt(orginal_prompt, upper=0.85, lower=0.65, max_tries=10):
    eles = word_tokenize(orginal_prompt)
    for i in range(max_tries):
        idx = randint(0, len(eles) - 1)
        chosen = lemmatizer.lemmatize(eles[idx])
        if chosen in nouns_set:
            eles[idx] = choice(nouns_list)
        elif chosen in adjs_set:
            eles[idx] = choice(adjs_list)
        elif chosen in verbs_set:
            eles[idx] = choice(verbs_list)
        else:
            continue
        candidate = ' '.join(eles)
        cs = semantic_similarity(orginal_prompt, candidate)
        if lower <= cs <= upper:
            return candidate
    return None


if __name__ == "__main__":
    prompt = 'i love  eat ice-cream, are you?'
    print(gen_a_new_prompt(prompt))
