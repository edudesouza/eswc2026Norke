
import time,asyncio,json

from rich           import print
from elasticsearch  import Elasticsearch

from src.utils      import diff_time, normalize
from src.config     import settings
from src.ingest     import graph_ingest

async def run_batch():
    
    print( '\n--- inicio ---') 
    print( f'--- Repo: {settings.repositorio} ---\n') 

      
    id   = 1
    _text = "Art. 7º O tratamento de dados pessoais somente poderá ser realizado nas seguintes hipóteses:\n\nI - mediante o fornecimento de consentimento pelo titular;\n\nII - para o cumprimento de obrigação legal ou regulatória pelo controlador;\n\nIII - pela administração pública, para o tratamento e uso compartilhado de dados necessários à execução de políticas públicas previstas em leis e regulamentos ou respaldadas em contratos, convênios\n\nou instrumentos congêneres, observadas as disposições do Capítulo IV desta Lei;\n\nIV - para a realização de estudos por órgão de pesquisa, garantida, sempre que possível, a\n\nanonimização dos dados pessoais;\n\nV - quando necessário para a execução de contrato ou de procedimentos preliminares\n\nrelacionados a contrato do qual seja parte o titular, a pedido do titular dos dados;\n\nVI - para o exercício regular de direitos em processo judicial, administrativo ou arbitral, esse\n\núltimo nos termos da Lei nº 9.307, de 23 de setembro de 1996 (Lei de Arbitragem);\n\nVII - para a proteção da vida ou da incolumidade física do titular ou de terceiro;\n\nVIII - para a tutela da saúde, em procedimento realizado por profissionais da área da saúde ou\n\npor entidades sanitárias;\n\nIX - quando necessário para atender aos interesses legítimos do controlador ou de terceiro,\n\nexceto no caso de prevalecerem direitos e liberdades fundamentais do titular que exijam a proteção dos\n\ndados pessoais; ou\n\nX - para a proteção do crédito, inclusive quanto ao disposto na legislação pertinente."
    text = '''Art. 18 O titular dos dados pessoais tem direito a obter do controlador, 
    em relação aos dados do titular por ele tratados, a qualquer momento e mediante 
    requisição:\n\n
    I - confirmação da existência de tratamento;\n\n
    II - acesso aos dados;\n\n
    III - correção de dados incompletos, inexatos ou desatualizados;\n\n
    IV - anonimização, bloqueio ou eliminação de dados desnecessários, excessivos ou tratados em desconformidade com o disposto nesta Lei;\n\n
    V - portabilidade dos dados a outro fornecedor de serviço ou produto, mediante requisição expressa e observados os segredos comercial e industrial, de acordo com a regulamentação do órgão\n\ncontrolador;\n\n
    VI - eliminação dos dados pessoais tratados com o consentimento do titular, exceto nas hipóteses previstas no art. 16 desta Lei;\n\nVII - informação das entidades públicas e privadas com as quais o controlador realizou uso compartilhado de dados;\n\n
    VIII - informação sobre a possibilidade de não fornecer consentimento e sobre as\n\nconsequências da negativa;\n\nIX - revogação do consentimento, nos termos do § 5º do art. 8º desta Lei.'''
    _text = "Art. 48 O controlador deverá comunicar à autoridade nacional e ao titular a ocorrência de incidente de segurança que possa acarretar risco ou dano relevante aos titulares."
    _text = "Art. 50 Os controladores e operadores, no âmbito de suas competências, pelo tratamento de dados pessoais, individualmente ou por meio de associações, poderão formular regras de boas práticas e\n\nde governança que estabeleçam as condições de organização, o regime de funcionamento, os procedimentos, incluindo reclamações e petições de titulares, as normas de segurança, os padrões técnicos, as obrigações específicas para os diversos envolvidos no tratamento, as ações educativas, os mecanismos internos de supervisão e de mitigação de riscos e outros aspectos relacionados ao\n\ntratamento de dados pessoais.Art. 50 Os controladores e operadores, no âmbito de suas competências, pelo tratamento de dados pessoais, individualmente ou por meio de associações, poderão formular regras de boas práticas e\n\nde governança que estabeleçam as condições de organização, o regime de funcionamento, os procedimentos, incluindo reclamações e petições de titulares, as normas de segurança, os padrões técnicos, as obrigações específicas para os diversos envolvidos no tratamento, as ações educativas, os mecanismos internos de supervisão e de mitigação de riscos e outros aspectos relacionados ao\n\ntratamento de dados pessoais.§ 1º Ao estabelecer regras de boas práticas, o controlador e o operador levarão em consideração, em relação ao tratamento e aos dados, a natureza, o escopo, a finalidade e a probabilidade e a gravidade dos riscos e dos benefícios decorrentes de tratamento de dados do titular.§ 2º Na aplicação dos princípios indicados nos incisos VII e VIII do **caput** do art. 6º desta Lei, o controlador, observados a estrutura, a escala e o volume de suas operações, bem como a sensibilidade\n\ndos dados tratados e a probabilidade e a gravidade dos danos para os titulares dos dados, poderá:\n\nI - implementar programa de governança em privacidade que, no mínimo:\n\na) demonstre o comprometimento do controlador em adotar processos e políticas internas que assegurem o cumprimento, de forma abrangente, de normas e boas práticas relativas à proteção de dados pessoais;\n\nb) seja aplicável a todo o conjunto de dados pessoais que estejam sob seu controle,\n\nindependentemente do modo como se realizou sua coleta;\n\nc) seja adaptado à estrutura, à escala e ao volume de suas operações, bem como à sensibilidade dos dados tratados;\n\nd) estabeleça políticas e salvaguardas adequadas com base em processo de avaliação sistemática de impactos e riscos à privacidade;\n\ne) tenha o objetivo de estabelecer relação de confiança com o titular, por meio de atuação\n\ntransparente e que assegure mecanismos de participação do titular;\n\nf) esteja integrado a sua estrutura geral de governança e estabeleça e aplique mecanismos de supervisão internos e externos;\n\ng) conte com planos de resposta a incidentes e remediação; e\n\nh) seja atualizado constantemente com base em informações obtidas a partir de monitoramento contínuo e avaliações periódicas;\n\nII - demonstrar a efetividade de seu programa de governança em privacidade quando\n\napropriado e, em especial, a pedido da autoridade nacional ou de outra entidade responsável por promover o cumprimento de boas práticas ou códigos de conduta, os quais, de forma independente,\n\npromovam o cumprimento desta Lei."
    file = ''   

    data = {
        "id":id,
        "arquivo":file,
        "id_usuario":"5511993891773",
        "id_externo":749,
        "texto":text
        #"texto":"Realizar apresentação para clientes ou reunião de negócios"
    }     

    result = await graph_ingest(data,debug=True) 
    try:                
        print( f'-> res: {result}' ) 
    except Exception as erro:
        print( f'-> ERRO processar ({id})' )  
            


if __name__ == "__main__":   
    asyncio.run( run_batch() )
