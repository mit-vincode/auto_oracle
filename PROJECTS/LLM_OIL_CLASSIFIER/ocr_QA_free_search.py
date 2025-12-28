# -*- coding: utf-8 -*-
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import faiss
import torch
import platform
from datetime import datetime as DT
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from bootstrap import *

load_dotenv(find_dotenv())
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ==============================================================================
# 1. АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ УСТРОЙСТВА (GPU/CPU)
# ==============================================================================
if platform.system() != "Darwin":
    mps_availability = False
else:
    mps_availability = torch.backends.mps.is_available()

if torch.cuda.is_available():
    device = "cuda"
elif mps_availability:
    device = "mps"
else:
    device = "cpu"

print(f"--- СИСТЕМА: {platform.system()} | ДВИЖОК: {device} ---")

# ==============================================================================
# 2. ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ
# ==============================================================================
MODEL_PATH = root_path + '/DATA_CATALOGS/llm_models/gguf/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf'
path_rag_index = root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/oil_rag_index/'

E5 = SentenceTransformer("intfloat/multilingual-e5-base", device=device)
index = faiss.read_index(os.path.join(path_rag_index, "faiss.index"))
with open(os.path.join(path_rag_index, "meta.jsonl"), "r", encoding="utf-8") as f:
    meta = [json.loads(l) for l in f if l.strip()]

correct_url_goods = set(
    U24.data2Df_upload(root_path + '/PROJECTS/LLM_OIL_CLASSIFIER/data_in/alleya_lavr_to_json.xlsx')['url'].unique())

# Пул бесплатных моделей для облака
FREE_MODELS_POOL = ["openai/gpt-4o-mini",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemini-2.0-flash-exp:free",
    "mistralai/mistral-7b-instruct:free",

]


# ==============================================================================
# 3. ЛОГИКА ОЧИСТКИ И ПОИСКА (RAG)
# ==============================================================================
def clean_rag_context(contexts, query):
    """Жесткая фильтрация брендов для исключения галлюцинаций."""
    query_l = query.lower()
    # Если ищем Ладу, выкидываем иномарки из контекста
    if any(x in query_l for x in ["лада", "lada", "грант", "vesta", "веста"]):
        return [c for c in contexts if
                "honda" not in c.get('title', '').lower() and "toyota" not in c.get('title', '').lower()]
    return contexts


def normalize_terminology(text: str) -> str:
    replacements = {r'(?i)аисин': 'AISIN', r'(?i)масло мотроное': 'Масло моторное'}
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text)
    return text


def retrieve(query: str, k: int = 5):
    q_vec = E5.encode([f"query: {query}"], normalize_embeddings=True, show_progress_bar=False).astype("float32")
    D, I = index.search(q_vec, k * 7)
    hits, seen_urls = [], set()
    for idx in I[0]:
        if idx == -1: continue
        item = meta[int(idx)]
        url = item.get("url", "").strip()
        if url in correct_url_goods and url not in seen_urls:
            hits.append(item);
            seen_urls.add(url)
        if len(hits) >= k + 3: break
    return hits


# ==============================================================================
# 4. ГЕНЕРАЦИЯ ОТВЕТА
# ==============================================================================
def generate_Answer(question: str, llm):
    raw_contexts = retrieve(question, k=5)
    contexts = clean_rag_context(raw_contexts, question)[:5]
    context_str = "\n".join([f"ТОВАР: {c.get('title')}. ТЕХ.ДАННЫЕ: {c.get('answer', '')}" for c in contexts])

    system_instruction = (
        "Ты — профессиональный технический консультант по автомобильным жидкостям и маслам."
        "Твой ответ — строго структурированный список товаров с техническими данными."
        
        ""
        "ПРАВИЛА КОНТЕНТА:"
        "1. ЗАПРЕЩЕНО писать вступления, обоснования и самооправдания (вроде 'сверьтесь с инструкцией')"
        "2 ЗАПРЕЩЕНО писать слова «Заголовок», «Вёрстка», «Завершение», «Ответ на вопрос»"
        "3. ЗАПРЕЩЕНО выводить параметры со значениями 'не указано', 'нет данных', 'отсутсвует','n/a' или оставлять их пустыми; Если данных об артикуле или бренде нет — удаляй всю строку целиком."
        "4. ЗАПРЕЩЕНО придумывать (галлюцинировать) технические параметры."
        "5. ЗАПРЕЩЕНО использовать Markdown разметку вида '**', '###'\n\n"
        ""
        "ТРЕБОВАНИЯ К ФОРМАТУ:"
        "• АВТОМОБИЛЬ клиента из вопроса (по-английски): марка  - модель  - модификация  - тип топлива - год. Важно: только те параметры автомобиля, которые понятны из контекста. Важно: запрещено выводить названия и модификации автомобилей, которых нет в вопросе клиента. Важно: Если автомобиля в вопросе клиента вообще нет - эту строку выводить ЗАПРЕЩЕНО!"
        "• Название товара (масло, спецжидкость, автохимия). Важно: выводить только те товары, про которые спрашивал клиент. и по которым есть данные."
        "• Бренд и Артикул. Формат: 'Бренд: [Name], Артикул: [Code]'. Важно: выводить только при наличии точных данных."
        "• [Спецификации] — SAE, API, ACEA, OE-допуски."
        "• [Технические данные] — Заправочный объем, интервал замены, способ применения.\n"
  
        ""
        "\n\nВажно! В ответе выводи только те товары, которые рашают проблему клиента. Дословно следовать шаблону ниже 'ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА' – ЗАПРЕЩЕНО!\n"
        "ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:\n"
       "для Opel Astra H (L35, L48, L69) 1.4 л, бензин\n" 
       "Моторное масло:\n"
       "• Объём: 5.7 л\n"
       "• Спецификация: VW 508.00\n"
       "• Периодичность замены: каждые 15 000 км\n\n"
       "Тормозная жидкость:\n"
       "• Объём: 0.9 л\n"
       "• Спецификация: DOT-4\n"
       "• Периодичность замены: каждые 2 года\n\n"
       "Масло в АКПП\n"
       "• Объём: 8.5 л\n"
       "• Спецификация: ATF DW-1\n\n"
        
        "Если в базе нашлись URL – ссылки на карточки товаров или источник информации, выведи их списокм. Важно: выводи только те товары и ссылки, которые решают проблему клиента. Если в базе несколько вариантов подходящих товаров, выводи 2-3 наиболее релевантных (например, Original OE и качественный аналог). Важно: URL (ссылки) обрезать и модифицировать ЗАПРЕЩЕНО!\n\n"
        
        "Вёрстка: использовать списки для читаемости, разделять логические блоки по типам продуктов, в конце ответа дать краткое резюме\n"
        "Приоритет содержания: Если вопрос касается технологии, обслуживания или нюансов эксплуатации, заменяй товарную карточку на краткую техническую справку."
        "Ответ должен быть строго по существу, решать проблему клиента, без воды и без лишних предисловий.\n"
        "Формулировка должна быть полезной, лаконичной и законченной.\n"

    )

    max_t = min(max(400 + (len(context_str) // 5), 400), 900)
    answer_text = None

    # --- API TRY ---
    if OPENROUTER_API_KEY and len(OPENROUTER_API_KEY) > 10:
        for model_id in FREE_MODELS_POOL:
            try:
                print(f"--> Trying Cloud API: {model_id}...")
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": f"БАЗА:\n{context_str}\n\nВОПРОС: {question}"}
                    ],
                    max_tokens=max_t,
                    temperature=0.0,
                    timeout=10
                )
                answer_text = response.choices[0].message.content.strip()
                if answer_text and len(answer_text) > 15:
                    print(f"Ai-Used: API ({model_id})\n")
                    break
            except Exception as e:
                print(f"    [SKIP] {model_id}: {type(e).__name__}")

    # --- LOCAL FALLBACK ---
    if not answer_text:
        print(f"Ai-Used: Local (Mistral) | Engine: {device}")
        full_prompt = f"[INST] {system_instruction}\n\nБАЗА:\n{context_str}\n\nВОПРОС: {question} [/INST]"
        output = llm(full_prompt, max_tokens=max_t, temperature=0.1, stop=["[/INST]"], echo=False)
        answer_text = output["choices"][0]["text"].strip()

    # Финальная чистка
    answer_text = re.split(r'(?i)обоснование|примечание', answer_text)[0].strip()

    # Ссылки
    verified_links = [c.get('url') for c in contexts if
                      any(p.lower() in answer_text.lower() for p in re.findall(r'[a-zA-Z0-9]{4,}', c.get('title', '')))]
    if verified_links:
        answer_text += "\n\n**Рекомендуемые товары:**\n" + "\n".join(list(dict.fromkeys(verified_links))[:3])

    return {"answer": answer_text}


# ==============================================================================
# 5. RUN
# ==============================================================================
if __name__ == "__main__":
    # Параметр n_gpu_layers=-1 автоматически задействует MPS на Mac или CUDA на Win
    llm = Llama(model_path=MODEL_PATH, n_ctx=8192, n_gpu_layers=-1, verbose=False)

    test_q = [ "Как Бабе Яге подготовить еë дизельную ступу к холодному пуску?", 'Volvo s60 2010 масло трансмиссионное',
               'Range Rover Vogue 3.6 TDI 2009 масло моторное', "Масло в двигатель g4na",]
    #     "",
    #     "Подбери масло в ДВС лада гранта 2020, 1.6 л",
    #            "Каое масло попросил бы двигатель Chery Tiggo 8 pro в письме Деду Морозу?",
    #            'Почему снежинки все разные, а масло 0W-20 - одинаковые (спойлер -нет)',
    #
    #            'Как Золушке разморозить замки на автомобиле, чтобы успеть на новогодний балл?',
    #            'Сколько граммов антифриза нужно для праздничного настроения?'
    #
    #           "Changan Lamore подбери трансмиссию",
    #
    #           "Jetour x50 1.5",
    #           "Масло для мотора g4na",
    #           "Подбери масло для Прадо 2008 года",
    #           "Land Cruiser Prado 2008",
    #           "У меня не дизель",
    #           "Масло трансмиссионное Volvo s60 коробка aisin",
    #
    #           "Audi a7 масло в коробку dsg",
    #           "Масло в коробку dsg",
    #           "volvo s60 2010 масло трансмиссионное",
    #           "Toyota Camry 2015 какое моторное масло и сколько литров?",
    #           "Как удалить рекламную наклейку с кузова автомобиля?",
    #           "Чем почистить карбюратор?",
    #           "Чем почистить пятно от чая на диване?",
    #           "Нужны ли антикоррозийные присадки для защиты топливного бака?"
    #           ]

    for q in test_q:
        T1 = U24.tNow()
        res = generate_Answer(q, llm)
        print(f"\n\nQ: {q}\n\nA: {res['answer']}")

        print(f"delta Time = {U24.deltaTimeProcess(T1)}")
        print(f"\n\n{'=' * 60}")







