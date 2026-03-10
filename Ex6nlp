import spacy
base = spacy.load("en_core_web_sm")
text="""Mr.G.Mano is a highly skilled software engineer
who has been working efficiently at Infosys for the past ten years.
He completed his Master’s degree in CSE from Anna University
and is currently living in Virudhunagar."""

doc = base(text)
print("NOUN PHRASES (NP):")
for chunk in doc.noun_chunks:
  print(chunk.text)

print("\nVERB PHRASES (VP):")
for token in doc:
  if token.pos_ == "VERB":
     print(token.text)

print("\nPREPOSITIONAL PHRASES (PP):")
for token in doc:
  if token.pos_ == "ADP":
  print(" ".join([token.text] + [child.text for child in token.children]))

print("\nADJECTIVE PHRASES (ADJP):")
for token in doc:
  if token.pos_ == "ADJ":
  print(token.text)

print("\nADVERB PHRASES (ADVP):")
for token in doc:
  if token.pos_ == "ADV":
  print(token.text)

print("\nNAMED ENTITIES:")
for ent in doc.ents:
  print(ent.text, "->", ent.label_)
