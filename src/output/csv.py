
import csv,os

def csv_create(name,retrieval,question,complexity,saf,nli,sim,response,model):

    csv_file = name
    output_row = [ retrieval, question, f'{complexity:.2f}',f'{saf:.2f}', f'{nli:.2f}', f'{sim:.2f}', response, model ]

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:

        writer = csv.writer(f, quoting=csv.QUOTE_ALL, delimiter=';')
        
        if not file_exists:
            writer.writerow(['type', 'question', 'saf_score', 'nli_score', 'sim_score', 'respose_llm'])
        
        writer.writerow(output_row)

    return 'OK'