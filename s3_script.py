import boto3
from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import os
from botocore.exceptions import ClientError
from _Lib import _config


def access_s3(id, secret):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=id,
        aws_secret_access_key=secret,
        region_name="sa-east-1",
    )
    return s3


def processar_documento_alocacao_mensal():
    key_id = _config("s3_aws_access_key_id")
    key_secret = _config("s3_aws_secret_access_key")
    s3 = access_s3(key_id, key_secret)

    # Nome do bucket e pasta
    bucket_name = "investimentos"
    folder_path = "carta_mensal_/"

    try:
        # Listar os objetos na pasta específica
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_path)
        if "Contents" not in response:
            print("Nenhum arquivo encontrado na pasta especificada.")
            return None

        # Filtrar apenas arquivos com extensão '.pdf'
        pdf_files = [
            obj for obj in response["Contents"] if obj["Key"].lower().endswith(".pdf")
        ]

        if not pdf_files:
            print("Nenhum arquivo PDF encontrado na pasta especificada.")
            return None

        # Ordenar os arquivos pela data de modificação (do mais recente para o mais antigo)
        pdf_files.sort(key=lambda x: x["LastModified"], reverse=True)

        # Selecionar o arquivo mais recente
        latest_pdf = pdf_files[0]
        object_key = latest_pdf["Key"]
        print(f"Processando o arquivo: {object_key}")

        # Baixar o arquivo
        response = s3.get_object(Bucket=bucket_name, Key=object_key)
        content = response["Body"].read()

        # Salvar o conteúdo em um arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Processar o PDF
        loader = PyMuPDFLoader(tmp_file_path)
        docs_carteira_mensal = loader.load()

        # Remover o arquivo temporário
        os.remove(tmp_file_path)

        return docs_carteira_mensal

    except ClientError as e:
        print(f"Erro ao acessar o bucket S3: {e}")
        return None
