from stanfordcorenlp import StanfordCoreNLP


# Preset
host = 'http://localhost'
port = 9000
nlp = StanfordCoreNLP(host, port=port,timeout=30000)

# The sentence you want to parse
sentence = 'The dog saw John in the park.'
print(sentence)
# POS
print('POS：', nlp.pos_tag(sentence))

# Tokenize
print('Tokenize：', nlp.word_tokenize(sentence))

# NER
print('NER：', nlp.ner(sentence))

# Parser
print('Parser：')
print(nlp.parse(sentence))
print(nlp.dependency_parse(sentence))

# The sentence you want to parse
sentence1 = 'The little bear saw the fine fat trout in the rocky brook.'
print(sentence1);
# POS
print('POS：', nlp.pos_tag(sentence1))

# Tokenize
print('Tokenize：', nlp.word_tokenize(sentence1))

# NER
print('NER：', nlp.ner(sentence1))

# Parser
print('Parser：')
print(nlp.parse(sentence1))
print(nlp.dependency_parse(sentence1))
# open and read from a file
f = open("demofile.txt", "r")
for x in f:
    print(x)
    # POS
    print('POS：', nlp.pos_tag(x))

    # Tokenize
    print('Tokenize：', nlp.word_tokenize(x))

    # NER
    print('NER：', nlp.ner(x))

    # Parser
    print('Parser：')
    print(nlp.parse(x))
    print(nlp.dependency_parse(x))

# Close Stanford Parser
nlp.close()