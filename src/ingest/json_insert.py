import json
from rich import print

with open("output/chunks_normativos.json", encoding="utf-8") as f:
    json_chunks = json.loads(f.read())

chunks = [item for item in json_chunks if item.get("tipo") == "chunk"]
total = len(chunks)

print( total )

'''for index, item in enumerate(json_chunks,start=1): 
    if item['tipo']=='chunk':
        print( item['id'])
        print( item['text'])'''

'''for item in json_chunks:
    if item['tipo']=='full_text':
        print( item['parent_id'])
        print( item['text'])'''