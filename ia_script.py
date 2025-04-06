from qdrant_client import QdrantClient
from langchain.schema import Document
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate
)
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from qdrant_script import chamar_collection
import textwrap
import os
import dotenv
import logging
from _Lib import _config

dotenv.load_dotenv()

# Carregar variáveis de ambiente
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
    # Recuperar pontos relevantes da coleção especificada
    if tipo == 'g':
        resultados = chamar_collection(collection_name, consulta_texto)
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
    if tipo == 'pd':
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


def analise_carteira(collection_name_xperformance, collection_name_mensal):
    llm = ChatOpenAI(model="gpt-4o", temperature=0.625)

    consulta_xperformance = "Relatório de performance da carteira do cliente"
    mensal_alocacao = "Relatório de recomendação de alocação de ativos"

    # Obter documentos das coleções
    xperformance_docs = obter_documentos_qdrant(
        collection_name_xperformance, consulta_xperformance, 'g'
    )
    mensal_docs = obter_documentos_qdrant(collection_name_mensal, mensal_alocacao, 'g')

    xperformance_text = "\n\n".join(
        [doc.page_content for doc in xperformance_docs if doc.page_content]
    )
    mensal_text = "\n\n".join(
        [doc.page_content for doc in mensal_docs if doc.page_content]
    )

    # Definir o esquema de resposta
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
                "Ex: Perfil: (tipo perfil)\n - Pós Fixado: x% - R$ y"
            ),
        ),
        ResponseSchema(
            name="comparacao_e_analise",
            description="Um parágrafo com uma comparação detalhada entre a alocação atual e a recomendada, identificando diferenças chave e analisando o impacto em termos de risco e retorno de forma bem desenvolvida e estruturada.",
        ),
        ResponseSchema(
            name="recomendacoes_para_rebalanceamento",
            description=(
                "Forneça recomendações específicas e diretas sobre como ajustar a carteira para alinhar-se à alocação recomendada. "
                "As recomendações devem ser apresentadas de forma numérica e organizada, indicando claramente quais ativos ou classes de ativos devem ser aumentados ou reduzidos, "
                "com percentuais e valores correspondentes. "
                "Proponha um plano de rebalanceamento gradual realizado em ondas, detalhando as etapas a serem tomadas. "
                "Separe cada recomendação por uma quebra de linha para facilitar a leitura."
                "Exemplo:\n"
                "[B]1ª Onda:[/B]\n"
                "- Reduzir exposição em **Renda Variável Brasil** de 28,17% (R$ 168.208,10) para 20% (R$ 120.000,00), movimentação: [B]R$ 48.208,1[/B].\n"
                "- Justificativa: O mercado doméstico está em alta volatilidade devido à instabilidade fiscal. Rebalancear para ativos mais defensivos pode reduzir o risco da carteira.\n"
                "- Timing: Realizar vendas gradualmente ao longo dos próximos 3 meses, aproveitando janelas de alta no mercado.\n\n"
                "[B]2ª Onda:[/B]\n"
                "- ..."
            )
        ),
        # ResponseSchema(
        #     name="call_de_saida",
        #     description="Forneça uma 'call de saída' com recomendações claras sobre quais ativos ativos ou classes de ativos o cliente deve considerar desinvestir,"
        #                 " incluindo o timing sugerido e as razões por trás dessas recomendações."
        #                 " Apresente as recomendações de forma estruturada e separada por quebras de linha para facilitar a leitura."
        #                 "Separe cada informação por uma quebra de linha para facilitar a leitura.\n"
        #                 "Ex: Perfil: (tipo perfil)\n - Pós Fixado: x% - R$ y"
        # )
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
        "2. Identifique se a carteira do cliente está alinhada com a recomendação de alocação em termos de exposição a diferentes classes de ativos (como renda fixa, renda variável, investimentos alternativos, etc.).\n"
        "3. Ofereça insights e recomendações **específicas, numéricas e organizadas** sobre como rebalancear a carteira, propondo um plano de rebalanceamento gradual realizado em ondas ao invés de imediatamente, para garantir que ela esteja em conformidade com os objetivos de investimento do cliente, levando em consideração fatores como tolerância ao risco, horizonte de investimento e condições de mercado atuais.\n"
        "   - As recomendações devem incluir **percentuais e valores** a serem ajustados em cada classe de ativo.\n"
        "   - Apresente as recomendações separadas por ondas (etapas) e use quebras de linha para facilitar a leitura.\n"
        "   - Exemplo de formato:\n"
        "     **1ª Onda:**\n"
        "     - Reduzir Renda Variável Brasil de 28,17% (R$ 168.208,10) para 22% (R$ 131.714,75)\n"
        "     - Aumentar Inflação de 10,89% (R$ 65.024,68) para 18% (R$ 107.492,46)\n"
        "     **2ª Onda:**\n"
        "     - ...\n"
        # "5. Forneça uma 'call de saída' com recomendações claras sobre quais ativos ou classes de ativos o cliente deve considerar desinvestir, incluindo o timing sugerido e as razões por trás dessas recomendações.\n"
        # "   - As recomendações devem ser específicas e levar em conta o cenário atual de mercado e os objetivos do cliente.\n"
        # "   - Apresente as recomendações separadas por quebras de linha para facilitar a leitura.\n"
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

    # Criar a cadeia de execução
    chain = prompt | llm | output_parser

    resposta = chain.invoke(
        {
            "xperformance_relevante": xperformance_text,
            "alocacao_relevante": mensal_text,
            "format_instructions": format_instructions,
        }
    )

    return resposta


def formatacao_resposta(resposta):
    for key in [
        "contextualizacao",
        "comparacao_e_analise",
        "recomendacoes_para_rebalanceamento",
        "oportunidades_de_melhoria",
    ]:
        if key in resposta and not isinstance(resposta[key], str):
            resposta[key] = (
                " ".join(resposta[key])
                if isinstance(resposta[key], list)
                else str(resposta[key])
            )

    alocacao_atual = resposta.get('alocacao_atual', '').replace(', ', '\n')
    alocacao_recomendada = resposta.get('alocacao_recomendada', '').replace(', ', '\n')
    recomendacoes_ondas = resposta.get('recomendacoes_para_rebalanceamento', '').replace(', ', '\n')

    resposta_formatada = (
            "[SIZE=4][B]Análise da carteira do cliente:[/B][/SIZE]\n\n\n"
            "[SIZE=3][B]Cenário de Investimentos:[/B][/SIZE]\n" + resposta.get('contextualizacao', '') + "\n\n"
            "[SIZE=3][B]Alocação Atual:[/B][/SIZE]\n" + alocacao_atual + "\n\n"
            "[SIZE=3][B]Alocação Recomendada:[/B][/SIZE]\n" + alocacao_recomendada + "\n\n"
            "[SIZE=3][B]Comparação e Análise:[/B][/SIZE]\n" + resposta.get('comparacao_e_analise', '') + "\n\n"
            "[SIZE=3][B]Recomendações para Rebalanceamento:[/B][/SIZE]\n" + recomendacoes_ondas + "\n\n"
    )

    return resposta_formatada


def main(collection_names):
    # collection_names é uma tupla com os nomes das coleções (xperformance, mensal)
    collection_name_xperformance, collection_name_mensal = collection_names

    resposta = analise_carteira(
        collection_name_xperformance, collection_name_mensal
    )

    resposta_formatada = formatacao_resposta(resposta)
    print(resposta_formatada)
    return resposta_formatada
