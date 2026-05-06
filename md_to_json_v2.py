import re
import json

def convert_md_to_json(md_file_path, json_file_path):
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # O padrão para este arquivo é: 
    # Pergunta: **Número. Pergunta**
    # Resposta: Texto logo abaixo da pergunta
    # O pattern busca por **Número. Pergunta** e captura tudo até a próxima pergunta ou separador ---
    pattern = re.compile(r'\*\*(\d+)\.\s+(.*?)\*\*\n(.*?)(?=\n\*\*\s*\d+\.|\n---|\Z)', re.DOTALL)
    
    matches = pattern.findall(content)
    
    data = []
    for match in matches:
        num, question, answer = match
        data.append({
            "id": int(num.strip()),
            "question": question.strip(),
            "answer": answer.strip()
        })

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Sucesso! {len(data)} perguntas convertidas e salvas em {json_file_path}")

if __name__ == "__main__":
    input_md = '100_perguntas_lgpd.md'
    output_json = '100_perguntas_lgpd.json'
    convert_md_to_json(input_md, output_json)
