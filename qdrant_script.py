import os
import re
import time
import numpy as np
from datetime import date, datetime
from pathlib import Path
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from s3_script import processar_documento_alocacao_mensal
from _Lib import _config
import dotenv

dotenv.load_dotenv()
qdrant_url = os.getenv("QDRANTURL")

openai_api_key = _config("OPENAI_KEY")
qdrant_api_key = _config("QDRANT_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
# Inicializar os embeddings e o cliente Qdrant
embeddings = OpenAIEmbeddings()
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


def sanitize_collection_name(name):
    # Remove caracteres que não sejam letras, números, underscores ou hifens
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", name)


def verificar_arquivo(caminho_pdf: str):
    if not os.path.exists(caminho_pdf):
        print(f"O arquivo '{caminho_pdf}' não foi encontrado.")
        return False
    return True


def sobrepor_mensal(collection_name_antigo):
    if collection_name_antigo.split("_")[-1] == "mensal":
        try:
            qdrant_client.delete_collection(collection_name_antigo)
            print(f"Coleção '{collection_name_antigo}' deletada com sucesso.")
        except Exception as e:
            print(f"Erro ao deletar coleção '{collection_name_antigo}': {e}")


def carregar_e_vetorizar_documento_xperformance(caminho_pdf: str):
    try:
        loader = PyMuPDFLoader(caminho_pdf)
        documentos = loader.load()
        textos = [doc.page_content for doc in documentos]
        vetores = embeddings.embed_documents(textos)
        if not vetores or any(v is None for v in vetores):
            print(f"Erro ao gerar os vetores para o documento '{caminho_pdf}'.")
            return None
        print(f"Gerados {len(vetores)} vetores para o documento '{caminho_pdf}'.")
        return textos, vetores
    except Exception as e:
        print(f"Erro ao carregar e vetorizar o documento '{caminho_pdf}': {e}")
        return None


def carregar_e_vetorizar_documento_mensal():
    try:
        documentos = processar_documento_alocacao_mensal()
        if not documentos:
            print("Nenhum documento carregado para carteira mensal.")
            return None
        textos = [doc.page_content for doc in documentos]
        vetores = embeddings.embed_documents(textos)
        if not vetores or any(v is None for v in vetores):
            print("Erro ao gerar os vetores para carteira mensal.")
            return None
        print(f"Gerados {len(vetores)} vetores para carteira mensal.")
        return textos, vetores
    except Exception as e:
        print(f"Erro ao carregar e vetorizar o documento mensal: {e}")
        return None


def criar_colecao_se_nao_existir(collection_name: str, vetor_exemplo):
    collection_name = sanitize_collection_name(collection_name)
    try:
        qdrant_client.get_collection(collection_name)
        print(f"A coleção '{collection_name}' já existe.")
    except UnexpectedResponse as e:
        if e.status_code == 404:
            if not qdrant_client.collection_exists(collection_name=collection_name):
                qdrant_client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qdrant_models.VectorParams(
                        size=len(vetor_exemplo), distance="Cosine"
                    ),
                )
                print(f"Coleção '{collection_name}' criada.")
        else:
            print(f"Erro ao verificar a coleção: {e}")
            return False
    return True


def inserir_vetores_na_colecao(collection_name: str, vetores, titulo: str, textos):
    try:
        pontos = []
        id_counter = 1

        # Se o título é "carteira_mensal", sobrepor qualquer coleção existente com "_mensal"
        if titulo == "carteira_mensal":
            collections = qdrant_client.get_collections()
            for collection in collections.collections:
                sobrepor_mensal(collection.name)
            time.sleep(1)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=len(vetores[0]), distance="Cosine"
                ),
            )

        for texto, vetor in zip(textos, vetores):
            pontos.append(
                qdrant_models.PointStruct(
                    id=id_counter,
                    vector=vetor,
                    payload={"title": titulo, "texto": texto},
                )
            )
            id_counter += 1

        # Inserir ou atualizar pontos na coleção
        qdrant_client.upsert(collection_name=collection_name, points=pontos)
        print(f"Documento '{titulo}' armazenado com sucesso em '{collection_name}'.")
    except Exception as e:
        print(f"Erro ao inserir dados na coleção '{collection_name}': {e}")


def armazenar_pdf_no_qdrant(caminho_pdf: str, collection_name_base="pdf_collection"):
    if not verificar_arquivo(caminho_pdf):
        return

    # Processar xperformance
    textos_xperformance, vetores_xperformance = (
        carregar_e_vetorizar_documento_xperformance(caminho_pdf)
    )
    if textos_xperformance is None or vetores_xperformance is None:
        return

    collection_name_xperformance = f"{collection_name_base}_xperformance"
    if not criar_colecao_se_nao_existir(
        collection_name_xperformance, vetores_xperformance[0]
    ):
        return

    inserir_vetores_na_colecao(
        collection_name_xperformance,
        vetores_xperformance,
        "xperformance",
        textos_xperformance,
    )

    # Processar documento mensal (permanece o mesmo)
    textos_mensal, vetores_mensal = carregar_e_vetorizar_documento_mensal()
    if vetores_mensal is None:
        return

    collection_name_mensal = f"{collection_name_base}_mensal"
    if not criar_colecao_se_nao_existir(collection_name_mensal, vetores_mensal[0]):
        return

    inserir_vetores_na_colecao(
        collection_name_mensal, vetores_mensal, "carteira_mensal", textos_mensal
    )

    print(
        f"Documentos retornados: {collection_name_xperformance}, {collection_name_mensal}"
    )
    return collection_name_xperformance, collection_name_mensal


def pegar_ultimo_pdf(user_id: str, caminho: str, deal_id: str):
    arquivos = [arquivo for arquivo in Path(caminho).glob(f"*{deal_id}*")]
    if not arquivos:
        print(
            f"Arquivo do user_id: '{user_id}' com deal_id: '{deal_id}' não encontrado em '{caminho}'."
        )
        return "", ""
    else:
        # Encontre o arquivo mais recente
        arquivoMaisRecente = max(arquivos, key=os.path.getmtime)
        nome_pdf = arquivoMaisRecente.stem
        return arquivoMaisRecente, nome_pdf


def extract_id_cliente(ultimo_caminho_pdf):
    pdf_path = ultimo_caminho_pdf
    try:
        # Verificar se o arquivo existe
        if not os.path.isfile(pdf_path):
            print(f"O arquivo {pdf_path} não foi encontrado.")
            return None

        # Inicializar o PyMuPDFLoader
        loader = PyMuPDFLoader(pdf_path)

        # Carregar o documento
        documentos = loader.load()

        if not documentos:
            print("Nenhum documento foi carregado.")
            return None

        # Extrair o texto da primeira página
        primeira_pagina = documentos[0].page_content if len(documentos) > 0 else ""
        if not primeira_pagina.strip():
            print("Nenhum texto foi extraído da primeira página.")
            return None

        # Dividir o texto por quebras de linha
        linhas = primeira_pagina.split('\n')

        # Inicializar variável para armazenar o ID do cliente
        id_cliente = None

        # Percorrer as linhas para encontrar "Data de Referência" e extrair o próximo valor
        for i, linha in enumerate(linhas):
            if "Data de Referência" in linha:
                # Verificar se existe uma linha seguinte
                if i + 1 < len(linhas):
                    possivel_id = linhas[i + 1].strip()
                    # Validar se a próxima linha é um número
                    if possivel_id.isdigit():
                        id_cliente = possivel_id
                        print(f"ID do Cliente encontrado: {id_cliente}")
                        return id_cliente
                    else:
                        print(f"O valor após 'Data de Referência' não é numérico: {possivel_id}")
                        return None

        # Caso não tenha encontrado "Data de Referência", tentar usar regex para encontrar um número de 6 dígitos
        if not id_cliente:
            pattern = r'\b(\d{6})\b'  # Padrão para encontrar exatamente 6 dígitos consecutivos
            match = re.search(pattern, primeira_pagina)
            if match:
                id_cliente = match.group(1)
                print(f"ID do Cliente encontrado via regex: {id_cliente}")
                return id_cliente
            else:
                print("ID do Cliente não encontrado na primeira página.")
                return None

    except Exception as e:
        print(f"Ocorreu um erro ao processar o PDF: {e}")
        return None


def salvar_collection(user_id: str, deal_id: str) -> tuple:
    caminho_pdf = os.path.join(os.getcwd(), "arquivos_salvos")
    ultimo_caminho_pdf, nome_pdf = pegar_ultimo_pdf(user_id, caminho_pdf, deal_id)

    if not ultimo_caminho_pdf:
        logging.error("Não foi possível encontrar o arquivo PDF.")
        return ()
    collection_names = ()
    try:
        # Sanitizar o nome da coleção
        id_cliente = extract_id_cliente(ultimo_caminho_pdf)
        collection_name_base = f"{nome_pdf}_{id_cliente}_{date.today().isoformat()}"
        collection_name_base = sanitize_collection_name(collection_name_base)

        collection_names = armazenar_pdf_no_qdrant(ultimo_caminho_pdf, collection_name_base)
        print(collection_names)
        # insert_xperformance(ultimo_caminho_pdf)
    except Exception as e:
        print(f"[ERRO] - Erro no processo de inserção da collection/no banco: {e}")
    return collection_names  # Retorna uma tupla com os nomes das coleções


def chamar_collection(collection_name: str, consulta_texto: str):
    try:
        # Vetorizar a consulta
        consulta_vetor = embeddings.embed_query(consulta_texto)

        # Realizar a busca semântica na coleção
        resultados = qdrant_client.search(
            collection_name=collection_name,
            query_vector=consulta_vetor,
            limit=20,
            with_payload=True,
            with_vectors=False,
        )

        print(
            f"Encontrados {len(resultados)} resultados na coleção '{collection_name}'."
        )
        return resultados  # Retornar os resultados para uso posterior

    except UnexpectedResponse as e:
        logging.error(f"Erro ao buscar a coleção '{collection_name}': {e}")
    except Exception as e:
        logging.exception(f"Erro inesperado na função 'chamar_collection': {e}")


def chamar_collection_posicao_detalhada(collection_name: str, consulta_texto: str):
    try:
        # Vetorizar a consulta
        consulta_vetor = embeddings.embed_query(consulta_texto)

        # Realizar a busca semântica na coleção
        resultados = qdrant_client.search(
            collection_name=collection_name,
            query_vector=consulta_vetor,
            limit=5,
            with_payload=True,
            with_vectors=False,
        )

        print(
            f"Encontrados {len(resultados)} resultados na coleção para pos detalhada '{collection_name}'."
        )
        return resultados  # Retornar os resultados para uso posterior

    except UnexpectedResponse as e:
        logging.error(f"Erro ao buscar a coleção detalhada '{collection_name}': {e}")
    except Exception as e:
        logging.exception(f"Erro inesperado na função 'chamar_collection_posicao_detalhada': {e}")
