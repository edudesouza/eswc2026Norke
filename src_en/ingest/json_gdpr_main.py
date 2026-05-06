import json

# python src_en/ingest/json_gdpr_main.py

with open("src_en/ingest/gdpr.json", encoding="utf-8") as f:
    json_gdpr = json.load(f)

for chapter in json_gdpr["chapters"]:
    
    for item in chapter["contents"]:
        articles = []

        if item["type"] == "article":
            articles.append(item)

        if item["type"] == "section":
            articles.extend(
                content
                for content in item["contents"]
                if content["type"] == "article"
            )

        for article in articles:
            id = f"chapter_{chapter['number']}_article_{article['number']}".lower()
            dados = f"Article {article['number']}: {article['title']}"

            for point in article["contents"]:
                
                number = point.get("number")
                text = point.get("text")

                if number and text:
                    dados += f"\n{number} {text}"
                elif text:
                    dados += f"\n{text}"

                for subpoint in point.get("subpoints", []):
                    sub_number = subpoint.get("number")
                    sub_text = subpoint.get("text")

                    if sub_number and sub_text:
                        dados += f"\n({sub_number}) {sub_text}"
                    elif sub_text:
                        dados += f"\n{sub_text}"

            print(id)
            print(dados)
            print('-'*100)
            
