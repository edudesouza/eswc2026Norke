import re
import json

def convert_md_to_json(md_file_path, json_file_path):
    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex para capturar a pergunta (### Número. Pergunta) e a resposta (**Resposta:** Texto)
    # O pattern busca por ### seguido de número, ponto e o texto da pergunta, 
    # e depois captura tudo até encontrar a próxima pergunta ou o fim do arquivo.
    pattern = re.compile(r'###\s+(\d+)\.\s+(.*?)\n\*\*Resposta:\*\*\s+(.*?)(?=\n###|\n---|\Z)', re.DOTALL)
    
    matches = pattern.findall(content)
    
    data = []
    for match in matches:
        num, question, answer = match
        data.append({
            "id": int(num),
            "question": question.strip(),
            "answer": answer.strip()
        })

    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Sucesso! {len(data)} perguntas convertidas e salvas em {json_file_path}")

if __name__ == "__main__":
    input_md = 'lgpd_perguntas_ambiguas_100.md'
    output_json = 'lgpd_perguntas_ambiguas_100.json'
    convert_md_to_json(input_md, output_json)
