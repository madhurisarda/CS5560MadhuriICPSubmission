import spacy
import textacy

nlp = spacy.load('en_core_web_sm')
text = 'CHICAGO (AP) —Citing high fuel prices, United Airlines said Friday it has increased fares by $6 per round trip on flights to some cities also served by lower-cost carriers. American Airlines, a unit AMR, immediately matched the move, spokesman Tim Wagner said. United, a unit of UAL, said the increase took effect Thursday night and applies to most routes where it competes against discount carriers, such as Chicago to Dallas and Atlanta and Denver to San Francisco, Los Angeles and New York.'

for sentence in text.split("."):
    val = nlp(sentence)
    tuples = textacy.extract.subject_verb_object_triples(val)
    print(tuples)
    tuples_list = []
    if tuples:
        tuples_to_list = list(tuples)
        tuples_list.append(tuples_to_list)
        print(tuples_list)
