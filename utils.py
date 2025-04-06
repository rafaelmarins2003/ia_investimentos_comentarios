import requests
from rich import print
import re
import os
from requests.auth import HTTPBasicAuth
from _Lib import _sql_update, _sql_select_valores_sql
from _clickhouse import ClickHouseConnection
from bitrix24 import Bitrix24
import dotenv

dotenv.load_dotenv()
bitrix_url = os.getenv("BITRIXURL")

bx24 = Bitrix24(bitrix_url)

async def crm_deal(deal_id):
    deals = await bx24.callMethod("crm.deal.list", filter={"ID": deal_id},
                            select=["ID", "ASSIGNED_BY_ID", "UF_CRM_1730832791461", "CONTACT_ID"])
    return deals


async def get_category_id_from_deal_id(deal_id: str) -> str:
    # Usa await para garantir que a tarefa seja aguardada e obtenha o resultado
    category_id = await bx24.callMethod("crm.deal.list", filter={"ID": deal_id}, select=["CATEGORY_ID"])

    # Verifica se `category_id` é uma lista e contém pelo menos um item
    if isinstance(category_id, list) and len(category_id) > 0:
        return str(category_id[0].get('CATEGORY_ID', ''))

    # Retorna o resultado bruto caso não seja possível acessar o índice [0]
    return str(category_id)


def add_timeline_comment(entity_id, comentario):
    url = "https://.bitrix24.com.br/rest/26/ln5dam49jf5wmpnv/crm.timeline.comment.add"

    data = {
        "fields": {"ENTITY_ID": entity_id, "ENTITY_TYPE": "deal", "COMMENT": comentario}
    }

    response = requests.post(url, json=data)
    if response.status_code == 200:
        response_data = response.json()
        if "error" in response_data:
            print(
                f"Erro (crm.timeline.comment.add): {response_data['error_description']}"
            )
        else:
            print(f"Comentário adicionado com sucesso. ID - {response_data['result']}")
    else:
        print(f"Erro na requisição (crm.timeline.comment.add): {response.status_code}")


def download_file(url, auth, filename):
    response = requests.get(url, auth=auth)
    if response.status_code == 200:
        with open(filename, "wb") as file:
            file.write(response.content)
        print(f"Arquivo {filename} baixado com sucesso.")
    else:
        print(
            f"Falha ao baixar o arquivo {filename}. Status code: {response.status_code}"
        )


async def baixar_pdf(deal_id):
    # Aguarde a execução da corrotina e depois acesse o índice
    deal = await crm_deal(deal_id)
    if deal and len(deal) > 0:
        deal = deal[0]
    else:
        print('Nenhum dado encontrado para o negócio.')
        return None, None, None, None

    if deal['UF_CRM_1730832791461'] is None:
        print('Arquivo não encontrado para download.')
        return None, None, None, None

    field = 'UF_CRM_1730832791461'
    file_id = deal['UF_CRM_1730832791461']['id']
    user_id = deal['ASSIGNED_BY_ID']
    contact_id = deal['CONTACT_ID']

    username = "rafael.marins@multisete.com"
    password = "12bc24AABB$"
    auth = HTTPBasicAuth(username, password)
    # Diretório para salvar os arquivos
    save_directory = os.path.join(os.getcwd(), "arquivos_salvos")
    os.makedirs(save_directory, exist_ok=True)

    # Tentar baixar o arquivo com autenticação
    file_url = f"https://.bitrix24.com.br/bitrix/components/bitrix/crm.deal.show/show_file.php?ownerId={deal_id}&fieldName={field}&fileId={file_id}"
    nome_pdf = f"{user_id}_{deal_id}.pdf"
    filename = os.path.join(save_directory, nome_pdf)
    download_file(file_url, auth, filename)
    return nome_pdf, deal_id, user_id, contact_id



def delete_all_files():
    folder_path = os.path.join(os.getcwd(), "arquivos_salvos")
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"{file_path} foi deletado.")
        except Exception as e:
            print(f"Erro ao deletar {file_path}: {e}")


def load_deals_memory():
    select = "SELECT valor FROM t_config WHERE campo = 'memoria_qdrant'"
    id_deals_antigos = _sql_select_valores_sql(select, "Digital")
    return id_deals_antigos[0]["valor"]


def save_deals_memory(deals_to_update):
    select = "SELECT valor FROM t_config WHERE campo = 'memoria_qdrant'"
    valor_antigo = _sql_select_valores_sql(select, "Digital")
    valores = ""
    for deal in deals_to_update:
        valores += f", {deal}"
    update = f"UPDATE t_config SET valor = '{valor_antigo[0]['valor']}{valores}' WHERE campo = 'memoria_qdrant'"
    return _sql_update(update, "Digital")


def insert_tb_hist_ia_investimentos(user_id, comentario, contact_id, collection_xperformance):
    select_nome = f"""
    SELECT 
        concat(JSONExtractString(json_data, 'NAME'), 
               if(JSONExtractString(json_data, 'LAST') != '', 
                  concat(' ', JSONExtractString(json_data, 'LAST')), 
                  '')) AS full_name
    FROM .t_historic_crm 
    WHERE entidade = 'user' 
      AND id_btx = '{user_id}'
    """
    user = ClickHouseConnection("").query(select_nome)
    if len(user) > 0:
        nome_user = user[0][0]
    else:
        nome_user = ''

    select_tipo = f"SELECT TOP 1 equipe FROM tb_estrutura WHERE crm_id = '{user_id}' ORDER BY Data_proc DESC"
    tipo = _sql_select_valores_sql(select_tipo, "Investimentos")
    if len(tipo) > 0:
        nome_tipo = tipo[0]['equipe']
    else:
        nome_tipo = ''

    select_nome_cliente = f"""
    SELECT
    concat(
        JSONExtractString(replaceRegexpAll(json_data, '[\\n\\r\\t]', ''), 'NAME'),
        if(
            JSONExtractString(replaceRegexpAll(json_data, '[\\n\\r\\t]', ''), 'LAST') != '',
            concat(' ', JSONExtractString(replaceRegexpAll(json_data, '[\\n\\r\\t]', ''), 'LAST')),
            ''
        )
    ) AS nome_completo
    FROM .t_historic_crm
    WHERE entidade = 'contact'
    """
    contact = ClickHouseConnection("").query(select_nome_cliente)
    if len(contact) > 0:
        nome_cliente = " ".join(word.capitalize() for word in contact[0][0].lower().split())
    else:
        nome_cliente = ''

    insert = f"""INSERT INTO tb_hist_ia_investimentos (id_btx, id_contact_btx, tipo, nome_user, resposta, dias, nome_contact, collection_xperformance)
                 VALUES ('{user_id}', '{contact_id}', '{nome_tipo}', '{nome_user}', '{comentario}', 30, '{nome_cliente}', '{collection_xperformance}');"""

    return _sql_update(insert, "Investimentos")
