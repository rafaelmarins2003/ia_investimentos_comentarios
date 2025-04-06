import os
import logging
import textwrap
from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from qdrant_script import chamar_collection, chamar_collection_posicao_detalhada
from _Lib import _config
import dotenv

dotenv.load_dotenv()

# Configurações iniciais
openai_api_key = _config("OPENAI_KEY")
qdrant_api_key = _config("QDRANT_API_KEY")

qdrant_url = os.getenv("QDRANTURL")

os.environ["OPENAI_API_KEY"] = openai_api_key
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_paragraph(text, width=80):
    return textwrap.fill(text, width=width)


def obter_documentos_qdrant(collection_name, consulta_texto, tipo):
    if tipo == 'g':
        resultados = chamar_collection(collection_name, consulta_texto)
    else:
        resultados = chamar_collection_posicao_detalhada(collection_name, consulta_texto)

    if not resultados:
        logging.error(f"Nenhum resultado encontrado na coleção '{collection_name}'.")
        return []
    docs = []
    for ponto in resultados:
        texto = ponto.payload.get("texto", "")
        doc = Document(
            page_content=texto,
            metadata={"payload": ponto.payload},
        )
        docs.append(doc)
    return docs


#############################################
# IA Inicial
#############################################

def analise_carteira(collection_name_xperformance, collection_name_mensal):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.625)

    consulta_xperformance = "Relatório de performance da carteira do cliente"
    mensal_alocacao = "Relatório de recomendação de alocação de ativos"

    xperformance_docs = obter_documentos_qdrant(collection_name_xperformance, consulta_xperformance, 'g')
    mensal_docs = obter_documentos_qdrant(collection_name_mensal, mensal_alocacao, 'g')

    xperformance_text = "\n\n".join(
        [doc.page_content for doc in xperformance_docs if doc.page_content]
    )
    mensal_text = "\n\n".join(
        [doc.page_content for doc in mensal_docs if doc.page_content]
    )

    # Schema original
    response_schema = [
        ResponseSchema(
            name="contextualizacao",
            description="Breve parágrafo contextualizando o cenário de investimentos de acordo com o relatório de recomendação de alocação de ativos estruturado.",
        ),
        ResponseSchema(
            name="alocacao_atual",
            description=(
                "Analise a alocação atual do cliente e compare-a com a alocação recomendada para identificar o tipo de perfil do cliente. "
                "Determine o perfil com base na proximidade dos percentuais atuais em relação aos percentuais recomendados, onde a menor diferença média entre os percentuais indica o perfil mais provável. "
                "Além disso, forneça um detalhamento claro e organizado da composição atual da carteira do cliente, incluindo **percentuais** e **valores investidos** em cada ativo ou classe de ativos. "
                "Apresente os resultados separados por quebras de linha para facilitar a leitura. "
                "Exemplo de formato:\n"
                "Perfil: (tipo de perfil)\n"
                "- Pós-Fixado: x% - R$ y\n"
                "- Renda Variável: x% - R$ y\n"
                "- Inflação: x% - R$ y"
            ),
        ),
        ResponseSchema(
            name="alocacao_recomendada",
            description=(
                "Apresente a alocação de ativos sugerida no relatório de recomendação, especificando os PERCENTUAIS e VALORES PROPOSTOS para cada classe de ativos. Siga estes passos:\n"
                "\n"
                "1. Identifique o perfil da carteira do cliente com base na alocação atual se comparado com o padrão identificado na carteira recomendada.\n"
                "2. Forneça as recomendações de alocação apropriadas para esse perfil baseada no relatório de recomendação.\n"
                "3. Separe cada informação por uma quebra de linha para facilitar a leitura.\n"
                "Ex: Perfil: (tipo perfil)\n - Pós-Fixado: x% - R$ y"
            ),
        ),
        ResponseSchema(
            name="comparacao_e_analise",
            description="Um parágrafo com uma comparação detalhada entre a alocação atual e a recomendada, identificando diferenças chave e analisando o impacto em termos de risco e retorno de forma bem desenvolvida e estruturada.",
        ),
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = (
        "Atue como um assessor de investimentos especializado em alocação de ativos e análise de performance de carteiras. "
        "Sua tarefa é analisar os seguintes relatórios e dados:\n\n"
        "1. **Relatório de Performance da Carteira do Cliente:**\n"
        "{xperformance_relevante}\n\n"
        "2. **Relatório de Recomendação de Alocação de Ativos:**\n"
        "{alocacao_relevante}\n\n"
        "Faça uma análise aprofundada e pouco genérica dos relatórios e conclua sua tarefa.\n\n"
        "Sua tarefa é:\n"
        "1. Analise a alocação atual da carteira do cliente conforme os dados de alocação atual e compare com as recomendações de alocação presentes no relatório de recomendação.\n"
        "Exemplo de Formato para alocação (sintético):\n"
        "Perfil: Conservador"
        "- Pós Fixado: 73,00% - R$ 881.756,74"
        "- Inflação: 17,50% - R$ 211.382,86"
        "- Prefixado: 0,00% - R$ 0,00"
        "- Multimercados: 2,00% - R$ 24.157,69"
        "- Ações: 1,50% - R$ 18.118,27"
        "- Ativos Internacionais: 5,00% - R$ 60.393,22"
        "- Alternativos: 1,00% - R$ 12.078,85"
        "2. Identifique se a carteira do cliente está alinhada com a recomendação de alocação em termos de exposição a diferentes classes de ativos (como renda fixa, renda variável, investimentos alternativos, etc.).\n"
        "OBS:\n"
        "- Tenha certeza de que os valores nos quais você está se baseando são valores reais existentes nos relatórios, e não inventados.\n"
        "- Lembre-se de que o rebalanceamento da carteira não deve ocorrer de forma imediata, mas sim ser realizado gradualmente em ondas.\n\n"
        "{format_instructions}\n\n"
        "Certifique-se de fornecer uma análise clara e fundamentada, utilizando linguagem acessível ao cliente, sem jargões técnicos excessivos.\n"
        "NÃO responda respostas genéricas sem relevância.\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(prompt_template),
        ]
    )

    chain = prompt | llm | output_parser
    resposta = chain.invoke(
        {
            "xperformance_relevante": xperformance_text,
            "alocacao_relevante": mensal_text,
            "format_instructions": format_instructions,
        }
    )

    return resposta


#############################################
# Agente de Recomendações de Rebalanceamento (Ondas)
#############################################
def agente_recomendacoes_ondas(collection_name_xperformance, collection_name_mensal, resposta_inicial):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    mensal_alocacao = "Relatório de recomendação de alocação de ativos"
    xperformance_texto = "Relatório de performance da carteira do cliente"

    xperformance_docs = obter_documentos_qdrant(collection_name_xperformance, xperformance_texto, 'g')
    mensal_docs = obter_documentos_qdrant(collection_name_mensal, mensal_alocacao, 'g')

    xperformance_text = "\n\n".join(
        [doc.page_content for doc in xperformance_docs if doc.page_content]
    )
    mensal_text = "\n\n".join(
        [doc.page_content for doc in mensal_docs if doc.page_content]
    )

    ondas_schema = [
        ResponseSchema(
            name="recomendacoes_para_rebalanceamento",
            description=(
                "Retorne apenas as recomendações de rebalanceamento em ondas, no formato solicitado. "
                "Cada onda deve apresentar todas as classes de ativos após o rebalanceamento, com percentuais e valores, totalizando exatamente 100%."
            )
        ),
    ]

    ondas_parser = StructuredOutputParser.from_response_schemas(ondas_schema)
    ondas_format_instructions = ondas_parser.get_format_instructions()

    prompt_template = (
        "Você é um agente especializado em recomendar rebalanceamentos em ondas. Sua tarefa é transformar as recomendações iniciais em um plano de rebalanceamento gradual, rigorosamente organizado em ondas.\n\n"

        "Recomendações Iniciais:\n{recomendacoes_iniciais}\n\n"
        "Relatórios disponíveis:\n"
        "1. **Relatório de Performance da Carteira do Cliente:**\n{xperformance_text}\n\n"
        "2. **Relatório de Recomendação de Alocação de Ativos:**\n{mensal_text}\n\n"

        "Instruções Detalhadas:\n"
        "1. Primeiro, calcule o valor total da carteira usando os dados disponíveis. Caso o valor total já esteja explícito, utilize-o. Caso não, deduza de forma coerente e explique brevemente como chegou a esse valor.\n"
        "2. Apresente as recomendações em uma ou mais ondas: [B]1ª Onda:[/B], [B]2ª Onda:[/B], etc.\n"
        "3. Cada onda representa um estado completo e final da carteira após o rebalanceamento daquela etapa.\n"
        "4. Liste todas as classes de ativos após cada onda, com percentuais e valores. A soma de todos os percentuais deve ser exatamente 100%. Não use aproximações, nem 99,99% nem 100,01%. Deve ser 100% exato.\n"
        "5. Ao fazer alterações:\n"
        "   - Indique a classe reduzida, mostrando percentual e valor antes e depois, além do valor movimentado.\n"
        "   - Indique a(s) classe(s) aumentada(s), mostrando percentual e valor antes e depois, além do valor movimentado.\n"
        "   - Se introduzir uma nova classe, reduza outras proporcionalmente. Se reduzir uma classe, aumente outras na mesma proporção.\n"
        "6. Inclua justificativa e timing para cada onda.\n"
        "7. Se for necessário, arredonde valores, mas sempre de forma coerente para chegar exatamente em 100%. Caso após os cálculos a soma não seja 100%, ajuste ligeiramente os percentuais (por exemplo, altere a última casa decimal) até obter 100% exato.\n"
        "8. Antes de apresentar a resposta final, verifique novamente seus cálculos. Se a soma não for 100%, revise os percentuais.\n"
        "9. Não invente valores fora do escopo fornecido. Se algum dado estiver faltando, explique como deduziu logicamente. Entretanto, mesmo deduzindo, você deve garantir a soma exata de 100%.\n"
        "10. Não use termos como 'aproximadamente' ou 'cerca de'. Seja específico e forneça valores exatos, mesmo que precise ajustá-los minimamente.\n"
        "11. Não forneça respostas genéricas. Seja específico, coerente e preciso.\n\n"

        "Exemplo de Formato (sintético):\n"
        "[B]1ª Onda:[/B]\n"
        "- Reduzir Renda Variável Brasil de 28,17% (R$ X) para 20% (R$ Y), movimentação: R$ Z.\n"
        "- Aumentar Ativos Internacionais de 2,15% (R$ A) para 10,15% (R$ B), movimentação: R$ C.\n"
        "- Manter Renda Fixa Brasil em 69,85% (R$ D).\n"
        "Distribuição final 1ª Onda (100%):\n"
        "- Renda Fixa Brasil: XX% (R$ XX)\n"
        "- Renda Variável Brasil: YY% (R$ YY)\n"
        "- Ativos Internacionais: ZZ% (R$ ZZ)\n"
        "Justificativa: ...\n"
        "Timing: ...\n\n"

        "OBS:\n"
        "- Verifique a soma final após cada onda. Deve ser exatamente 100%.\n"
        "- Ajuste percentuais o quanto for necessário até chegar a 100%.\n\n"

        "{format_instructions}\n\n"
        "NÃO responda com saídas genéricas. Seja preciso e coerente."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(prompt_template),
        ]
    )

    chain = prompt | llm | ondas_parser

    resposta_ondas = chain.invoke({
        "recomendacoes_iniciais": resposta_inicial,
        "xperformance_text": xperformance_text,
        "mensal_text": mensal_text,
        "format_instructions": ondas_format_instructions
    })

    return resposta_ondas['recomendacoes_para_rebalanceamento']


#############################################
# Agente de Call de Saída a Nível de Ativos
#############################################
def agente_call_de_saida(collection_name_xperformance, resposta_ondas):
    # Este agente recebe as ondas do agente anterior e produz o call de saída detalhado a nível de ativos
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    consulta_xperformance_detalhada = 'Posição detalhada dos ativos'
    xperformance_docs = obter_documentos_qdrant(collection_name_xperformance, consulta_xperformance_detalhada, 'pd')
    xperformance_detalhada_text = "\n\n".join(
        [doc.page_content for doc in xperformance_docs if doc.page_content]
    )

    # Schema para call_de_saida
    call_schema = [
        ResponseSchema(
            name='call_de_saida',
            description="Forneça uma 'call de saída' com recomendações claras e alinhadas às ondas."
        )
    ]

    call_parser = StructuredOutputParser.from_response_schemas(call_schema)
    call_format_instructions = call_parser.get_format_instructions()

    prompt_template = (
        "Você é um agente especializado em criar calls de saída a nível de ativo, alinhadas com as ondas de rebalanceamento definidas anteriormente.\n\n"
        "Abaixo seguem as recomendações definidas em ondas:\n{recomendacoes_ondas}\n\n"
        "Aqui está a posição detalhada da carteira do cliente:\n{xperformance_detalhada_text}\n\n"

        "Sua tarefa: Gerar uma 'call de saída' que detalhe, para cada onda, quais ativos individuais devem ser desinvestidos. Essas recomendações de desinvestimento precisam:\n"
        "- Estar numericamente alinhadas com as proporções e valores estabelecidos nas ondas anteriores.\n"
        "- Refletir os mesmos percentuais e valores definidos para cada classe de ativos nas ondas, garantindo que a soma dos desinvestimentos em determinados ativos corresponda às movimentações previstas.\n"
        "- Não inventar valores; utilizar apenas os dados da posição detalhada e as alocações definidas nas ondas.\n"
        "- Priorizar o desinvestimento dos ativos com menor performance (rendimentos inferiores), principalmente aqueles com maiores valores investidos, conforme as indicações já dadas nas ondas.\n\n"

        "Instruções detalhadas:\n"
        "1. Analise cada onda e as classes de ativos que devem ser reduzidas ou das quais se deve sair completamente.\n"
        "2. Verifique, na posição detalhada, quais ativos pertencem a essas classes e têm menor rendimento.\n"
        "3. Selecione os ativos a serem desinvestidos para atingir o valor e percentual de redução previstos em cada onda.\n"
        "4. A soma dos valores desinvestidos por ativo em cada onda deve corresponder exatamente (ou o mais próximo possível) ao valor total de desinvestimento previsto para a classe naquela onda.\n"
        "5. Caso não haja dados suficientes para um cálculo exato, ajuste a recomendação de forma coerente e justifique brevemente.\n"
        "6. Use quebras de linha para melhor legibilidade.\n\n"

        "{format_instructions}\n"
        "Exemplo de Formato (simplificado):\n"
        "[B]1ª Onda:[/B]\n"
        "- Reduzir CDBs com menor rentabilidade, priorizando:\n"
        "  - CDB BANCO VOITER S.A. - DEZ/2024: R$ 114.747,43\n"
        "  - CDB BMG - DEZ/2024: R$ 5.115,49\n"
        "(total desinvestido nesta onda: R$ 119.862,92, alinhado com o valor definido na onda)\n\n"
        "[B]2ª Onda:[/B]\n"
        "- Continuar a redução no Pós-Fixado, reduzindo:\n"
        "  - CDB BS2 - DEZ/2024: R$ 208.287,89\n"
        "  - CDB NEON FINANCEIRA - DEZ/2024: R$ 114.639,95\n"
        "  - CDB SENFF - JAN/2025: R$ 27.036,96\n"
        "  - LCI BANCO STELLANTIS - DEZ/2024: R$ 77.570,52\n"
        "(total reduzido nesta onda: R$ 427.535,32, alinhado ao valor definido na onda)\n\n"
        "Adapte o exemplo acima aos valores reais da carteira e às ondas definidas, garantindo o alinhamento numérico exato."
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(prompt_template),
        ]
    )

    chain = prompt | llm | call_parser

    resposta_call = chain.invoke({
        "recomendacoes_ondas": resposta_ondas,
        "xperformance_detalhada_text": xperformance_detalhada_text,
        "format_instructions": call_format_instructions
    })

    return resposta_call['call_de_saida']


#############################################
# Formatação Final da Resposta
#############################################

def formatacao_resposta(resposta_final):
    alocacao_atual = resposta_final.get('alocacao_atual', '').replace(', ', '\n')
    alocacao_recomendada = resposta_final.get('alocacao_recomendada', '').replace(', ', '\n')
    recomendacoes_ondas = resposta_final.get('recomendacoes_para_rebalanceamento', '').replace(', ', '\n')

    resposta_formatada = (
            "[SIZE=4][B]Análise da carteira do cliente:[/B][/SIZE]\n\n\n"
            "[SIZE=3][B]Cenário de Investimentos:[/B][/SIZE]\n" + resposta_final.get('contextualizacao', '') + "\n\n"
            "[SIZE=3][B]Alocação Atual:[/B][/SIZE]\n" + alocacao_atual + "\n\n"
            "[SIZE=3][B]Alocação Recomendada:[/B][/SIZE]\n" + alocacao_recomendada + "\n\n"
            "[SIZE=3][B]Comparação e Análise:[/B][/SIZE]\n" + resposta_final.get('comparacao_e_analise', '') + "\n\n"
            "[SIZE=3][B]Recomendações para Rebalanceamento:[/B][/SIZE]\n" + recomendacoes_ondas + "\n\n"
            "[SIZE=3][B]Call de Saída:[/B][/SIZE]\n" + resposta_final.get('call_de_saida', '') + "\n\n"
    )

    return resposta_formatada


#############################################
# Função Principal
#############################################
def main(collection_names):
    # IA inicial (sem agentes), produz a resposta original
    collection_name_xperformance, collection_name_mensal = collection_names
    resposta_inicial = analise_carteira(collection_name_xperformance, collection_name_mensal)

    # Agente de recomendações em ondas (recebe a resposta_inicial)
    novas_recomendacoes_ondas = agente_recomendacoes_ondas(collection_name_xperformance, collection_name_mensal,
                                                           resposta_inicial)

    # Agente call de saída (recebe as ondas)
    call_de_saida = agente_call_de_saida(collection_name_xperformance, novas_recomendacoes_ondas)

    # Montar a resposta final juntando tudo
    resposta_final = resposta_inicial.copy()
    resposta_final['recomendacoes_para_rebalanceamento'] = novas_recomendacoes_ondas
    resposta_final['call_de_saida'] = call_de_saida

    resposta_formatada = formatacao_resposta(resposta_final)
    print(resposta_formatada)
    return resposta_formatada

# Exemplo de chamada:
# main(("colecao_xperformance", "colecao_mensal"))
