from qdrant_script import salvar_collection
from ia_script_agents import main as ia
from fastapi import FastAPI, Request
from utils import *
import sys
import requests

app = FastAPI()

print("Inicializando o aplicativo...")
print(f"Versão do Python: {sys.version}")


async def process_deal_update(deal_id):
    try:
        category_id = await get_category_id_from_deal_id(str(deal_id))
        print(f'category_id: {category_id} | type: {type(category_id)}')
        if str(category_id) != 'correta':
            print("Categoria não é correta. Ignorando atualização.")
            return {"status": "sucesso", "mensagem": "Categoria não é correta."}
        deals_memory = load_deals_memory()
        deals_to_update_in_memory = []
        nome_pdf, deal_id, user_id, contact_id = await baixar_pdf(deal_id)

        if nome_pdf is None or deal_id is None or user_id is None:
            return {"status": "sucesso", "mensagem": "Deal sem arquivo."}

        if str(deal_id) in deals_memory:
            print(f"Deal: {deal_id} já analisado!")
            return {"status": "sucesso", "mensagem": "Sem arquivos."}

        collection_name = salvar_collection(user_id=user_id, deal_id=deal_id)
        deals_to_update_in_memory.append(str(deal_id))
        comentario = ia(collection_name)
        add_timeline_comment(deal_id, comentario)
        insert_tb_hist_ia_investimentos(user_id, comentario, contact_id, collection_name[0])
        if len(deals_to_update_in_memory) > 0:
            save_deals_memory(deals_to_update_in_memory)
        delete_all_files()
        return {"status": "sucesso", "mensagem": "Processamento concluido"}
    except requests.exceptions.RequestException as e:
        print(f"Erro ao processar arquivos. Erro: {e}")
        return {"status": "erro", "mensagem": f"Erro ao processar arquivos: {e}"}


@app.get("/")
async def root():
    return {"message": "Aplicativo em execução"}


@app.post("/webhook")
async def receive_webhook(request: Request):
    print("Webhook recebido")
    try:
        content_type = request.headers.get("Content-Type")
        if "application/json" in content_type:
            data = await request.json()
        else:
            data = await request.form()
            data = dict(data)

        deal_id = data.get("data[FIELDS][ID]") or data.get("data[ID]")
        print(f'deal_id: {str(deal_id)} | type: {type(deal_id)}')

        await process_deal_update(deal_id)


    except Exception as e:
        print(f"Erro ao processar o webhook: {e}")
        import traceback

        traceback.print_exc()

    return {"status": "success"}
