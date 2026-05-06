
import csv,os

def csv_create(name,id,retrieval,question,gold_answer,response,complexity,saf,nli,sim,relevancy,faithfulness,model):

    csv_file = name
    output_row = [
        id, 
        retrieval, 
        question, 
        gold_answer,
        response, 
        f'{complexity:.2f}',f'{saf:.2f}', f'{nli:.2f}', f'{sim:.2f}', f'{relevancy:.2f}', f'{faithfulness:.2f}',
        model 
    ]

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:

        writer = csv.writer(f, quoting=csv.QUOTE_ALL, delimiter=';')
        
        if not file_exists:
            writer.writerow([
                'id',
                'type', 
                'question', 
                'gold_answer',
                'llm_response',  
                'complexity','saf', 'nli', 'sim', 'relevancy', 'faithfulness',
                'model'
            ])
        
        writer.writerow(output_row)

    return 'OK'