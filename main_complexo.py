
import csv,os,sys,time,asyncio

from rich import print

from src.services   import keywords_create, graph_search, vector_search, response_create, ground_truth
from src.utils      import diff_time
from src.evaluation import saf, score_dynamic_gt
from src.config     import settings

async def main(user_id,pergunta,retrieval='grafo',retrieval_size=1,size_gt=5,debug_all=False,debug_one=[],output=False):  

    print( f'[red] \n--- inicio ---' )

    if not pergunta:
        print('Nenhuma pergunta econtrada!')
        exit()
    
    inicio = time.time()  
    inicio_global = time.time()    

    print( f'\nUSER: {pergunta}' )
    print( '-'*100,'\n' )

    # query expantion
    #--------------------------------------------------------------------------

    # gpt-4.1, settings.OPENAI_API_KEY
    # gemini-2.5-flash, settings.GEMINI_API_KEY
    expantion        = keywords_create(pergunta,'gemini-2.5-flash',settings.GEMINI_API_KEY)
    palavras_chave   = expantion['keywords']
    complexity_score = expantion['complexity_score']

    #palavras_chave = 'Churrasco, Uso de churrasqueira, Regulamento do condomínio, Permissão para churrasco'        

    # quanto maior a similaridade mais próximo a um tema único
    if complexity_score<0.75:
        print( f'-> complexidade alta: {complexity_score:.2f}' )
        retrieval_size = 10
        size_gt = 5
    else:
        print( f'-> complexidade baixa: {complexity_score:.2f}' )

    diff_time('\n-> #1 expandir query OK: ', inicio)

    if debug_all or 'query' in debug_one:
        print( f'[yellow]// Debug keywords:' )
        
        for kw_key,kw_value in expantion['query_expansion'].items():        
            if kw_key and kw_value not in ['NULL','null']: 
                print( f'[yellow]- {kw_value}' )
        print('\n')
    
    # retriever
    #--------------------------------------------------------------------------
    
    inicio = time.time() 

    contexto    = ''
    colecao_llm = ''
    knowledge   = {}

    for kw_key,kw_value in expantion['query_expansion'].items():
        
        if kw_value and str(kw_value).lower() != "null":  

            if retrieval=='grafo':

                recuperacao  = graph_search(kw_value,pergunta,user_id,retrieval_size)
                llm          = response_create(kw_value,pergunta,recuperacao,'gpt')
                colecao_llm += f'<pergunta>{kw_value}</pergunta>\n<resposta>{llm['resposta']}</resposta>\n'                
                contexto    += f'{kw_value}\n{recuperacao["response"]}'
                knowledge.update(recuperacao['dataset'])
                print( f'->LLM: {colecao_llm}')
            
            else:  

                recuperacao  = vector_search(kw_value,pergunta,'documentos',user_id,retrieval_size)
                llm          = response_create(kw_value,pergunta,recuperacao,'gpt')
                colecao_llm += f'<pergunta>{kw_value}</pergunta>\n<resposta>{llm['resposta']}</resposta>' 
                contexto    += f'{kw_value}\n{recuperacao["response"]}'
                knowledge.update(recuperacao['dataset']) 
                print( f'->LLM: {colecao_llm}') 

            diff_time('\n-> #2 buscar dados, OK: ', inicio)
            
            if debug_all or 'retriever' in debug_one:
                print( f'[yellow]// Debug contexto  {retrieval}:\n{recuperacao['response']}' )
                print( f'[yellow]// Debug knowledge {retrieval}:\n{recuperacao['dataset']}' )

    # ground truth and response
    #--------------------------------------------------------------------------

    inicio = time.time() 

    task_ground_truth           = asyncio.to_thread(ground_truth,contexto,pergunta,palavras_chave,'ollama',size_gt)
    task_response_llm           = asyncio.to_thread(response_create,palavras_chave,pergunta,colecao_llm,'ollama')
    response_gt, response_llm   = await asyncio.gather(task_ground_truth, task_response_llm)
    resposta                    = response_llm['resposta']

    print( f'\nLLM: {resposta}\n' )

    diff_time('\n-> #3 ground truth e resposta OK: ', inicio)

    if debug_all or 'ground_truth' in debug_one:
        print( f'[yellow]// Debug GT:\n{response_gt}\n' )

    # metrics
    #--------------------------------------------------------------------------
    
    inicio  = time.time()    
    
    saf_score                   = saf(knowledge,resposta,pergunta,debug_all)
    dyn_score                   = asyncio.to_thread( score_dynamic_gt,response_gt,resposta )
    response_saf, response_dyn  = await asyncio.gather(saf_score, dyn_score)

    nli_val   = response_dyn.get('score_nli', {}).get('score', 0)
    sim_val   = response_dyn.get('score_sim', {}).get('score', 0)
    match_txt = response_dyn['matched']    
    
    print( f'\n {match_txt}' )
    print( f"-> nli: {nli_val:.2f}" )
    print( f"-> sim: {sim_val:.2f}" )
    print( f'-> saf: {response_saf:.2f}' )
    
    diff_time('\n-> #4 factualidade e comparação: ', inicio)  

    if output==True:

        csv_file = "bench_grafo_0812.csv"
        output_row = [ retrieval, pergunta, f'{saf_score:.2f}', f'{nli_val:.2f}', f'{sim_val:.2f}', resposta ]

        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='', encoding='utf-8-sig') as f:
    
            writer = csv.writer(f, quoting=csv.QUOTE_ALL, delimiter=';')
            
            if not file_exists:
                writer.writerow(['tipo', 'chave', 'saf_score', 'nli_score', 'sim_score', 'resposta_llm'])
            
            writer.writerow(output_row)

    if( nli_val==0 and sim_val==0 and response_saf>0.5 ):

        print( '[red]*** Resposta com alto grau de ambiguidade: retriver, llm ou gt problemático ***\n' )

        #print( palavras_chave )
        #print( '-'*100 )
        #print( contexto )
        #print( '-'*100 )
        #print( response_gt )   
    
    diff_time('-> Tempo total: ', inicio_global) 
    print( f'[red] --- fim ---\n' )

if __name__ == "__main__":
    
    pergunta1  = 'posso parar duas motos na minha vaga?'
    pergunta2  = 'meu carro é pequeno e vi que se eu parar a minha moto dentro da minha vaga, cabe e não atrabalha ninguem, blz?'
    pergunta3  = 'oi, bom dia, eu preciso fazer uma apresentação para uns clientes e pensei em fazer no salão de festas é algo pequeno só 20 pessoas, posso?' 
    pergunta4  = 'Poderia me ajudar com uma dúvida sobre o fundo de obra ... para que eu preciso pagar se não está acontecendo nenhuma obra?' 
    pergunta5  = 'estou recebendo uns parentes aqui no meu apartamento e hoje está muito quente a gente pode ir para a piscina rapidinho?'
    pergunta6  = 'sou médico e só tenho o domingo livre, não existe forma alguma de fazer a minha mudança no próximo domingo? o síndico não pode aprovar essa exceção, ele me falou no elevador que por ele OK'
    pergunta7  = 'estou com a perna quebrada e é a segunda vez que vcs impendem meu ifood de ser entregue, eu não tenho como descer da próxima vez vou chamar a polícia!'
    pergunta8  = 'estou com a perna quebrada e vcs impendem meu ifood subir, não temo como abrir uma exceção?'
    pergunta9  = 'minha arquiteta sugeriu a aplicação de um sobre piso, disse que é rápido não afeta a carga e não precisa de ART, posso fazer?'
    pergunta10 = 'quero fazer um churrasco mas vi que a churrasqueira está ocupada, posso fazer um churrasco com um churrasqueira portátil lá perto do jardim, vi que o regulamento não proibe, concorda?'
    pergunta11 = 'vou passar 3 meses fora trabalhando em um outro estado e nesse período vou fazer um AirBnB aqui, vi o regulamento e a convenção e nenhum probe então entendo que está OK, blz?'
    pergunta12 = 'roubaram minha bike dentro do condomínio, isso é um absurso, o condomínio deve me reembolsar?'
    pergunta13 = 'roubaram o carro do meu filho em frente ao condomínio, quero as imagens agora e o síndico não quer me fornecer, pode isso?'
    pergunta14 = 'oi, bom dia, eu preciso fazer uma demonstração de produtos para meus clientes e pensei em alugar o salão de festas, serão umas 20 pessoas, OK?'
    pergunta15 = 'oi, bom dia, quero fazer um culto com os irmãos da igreja no próximo dia 10 e quero alugar o salão, OK?' 
    pergunta16 = 'porque meus convidados que estão no meu aniversário não podem fumar aqui na area de fora, perto da churrasqueira'
    pergunta17 = 'o síndico não quer me passar as imagens da camera de segurança, o carro do meu irmão foi roubado em frente ao condomínio, ele pode negar isso? ele falou que a lgpd não deixa!'
    pergunta18 = 'Alguém sabe qual é a diretriz do condomínio quando o carro tem o comprimento maior que a vaga (entrando na área de circulação)? O carro ao lado do meu está assim e estou com receio de encostar nele durante a manobra. Queria entender qual é o procedimento/orientação nesses casos. Não quero gerar multa pra ninguém mas fica bem pior pra manobrar.'
    pergunta19 = 'Falei hoje de manã com o síndico e ele me falou que está OK eu deixar o meu sofá antigo na garagem, por que vocês continuam me incomodando?'
    pergunta20 = 'Eu me recuso a me registrar na biometria facial, não existe nada legalmente que me obrigue a isso, certo?'
    pergunta21 = 'posso andar de skate na quadra?'

    '''
    Exemplo de pergunta com alto grau de ambiguidade: 
    fazer um culto de final de ano com os irmãos aqui do predio - regras sobre reserva e regra sobre evento em áera comum
    vi que o salão não está ocupado                             - chunk sobre procedimento de reserva
    como é só pessoal daqui mesmo, acho que não precisa pagar   - regra sobre reserva e chunk sobre valores de localção
    '''

    pergunta_debugger = "Pensei usar o salão que não está ocupado no próximo final de semana, para um culto de final de natal só com os moradores e como é só pessoal daqui mesmo, acho que não precisa pagar né? obrigado deus te abençõe!"

    if len(sys.argv) < 2:
        #print("Uso: digite a perguta, exemplo pergunta1 \"nr da pergunta aqui\"")
        #sys.exit(1)
        pergunta = pergunta_debugger
        banco    = 'grafo'        
    else:
        banco    = sys.argv[1].strip()
        pergunta = globals().get( sys.argv[2].strip() )

    asyncio.run( main('5511993891773',pergunta,banco,10,5,False,['query'],False) )
