import os

from factscore.factscorer import FactScorer

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

fs = FactScorer(openai_key=OPENAI_API_KEY)

topics = [
'''Página 12 de 43 Regulamento Interno CONDOMÍNIO ALTO DO IPIRANGA NOUVEAUX responsável, no caso 
de constatação de irregularidades; 87º - O condômino responsável pela irregularidade na reforma, será obrigado
a arcar com todas as despesas necessárias para reparar os danos causados à estrutura da construção, incluindo 
os custos relativos aos materiais, mão de obra e eventuais projetos técnicos exigidos; 88º - O condômino 
deverá efetuar o pagamento dessas despesas no prazo estabelecido pelo Condomínio, que fará a cobrança através 
dos) lançamento(s) no(s) boleto(s) para pagamento das colas condominiais, cuja forma de pagamento obedecerá a 
mesma negociada com os profissionais contratados quanto a valores e eventuais parcelamentos. Artigo 26º - 
Reparos, de caráter urgente e inadiável, poderão ser realizados em qualquer dia e hora, a critério do Síndico.
Em caso de aprovação deverão ser comunicados aos demais CONDÔMINOS que possam ser incomodados pela obra.''',

]

generations = ['Você precisa pagar o fundo de obra porque todos os condôminos são obrigados a contribuir para as despesas comuns do condomínio, incluindo o custeio de obras, mesmo que não haja obras em andamento atualmente.']

# topics: list of strings (human entities used to generate bios)
# generations: list of strings (model generations)
out = fs.get_score(topics, generations, gamma=10)

print (out["score"]) # FActScore
print (out["init_score"]) # FActScore w/o length penalty
print (out["respond_ratio"]) # % of responding (not abstaining from answering)
print (out["num_facts_per_response"]) # average number of atomic facts per response