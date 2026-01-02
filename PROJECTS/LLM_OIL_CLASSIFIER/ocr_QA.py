# -*- coding: utf-8 -*-
# ocr_QA.py = oil_classifier_Question_Answer
from bootstrap import *
from USEFUL_UTILS.ssDecoder_car_bart import ss_2_CarParams
from bootstrap import root_path
import os
import re
from llama_cpp import Llama
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

# Загрузка параметров авто
CarParam = ss_2_CarParams()

from PROJECTS.LLM_OIL_CLASSIFIER.osr_filter_attributes import filter_car_params
from PROJECTS.LLM_OIL_CLASSIFIER.ocr_QA_free_search import generate_Answer
from PROJECTS.LLM_OIL_CLASSIFIER.ocr_select_oil_assortment import select_Oil_assortment

# Настройка окружения
os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv(find_dotenv())
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

# ==============================================================================
# НАСТРОЙКИ МОДЕЛЕЙ
# ==============================================================================
model_dir = root_path + '/DATA_CATALOGS/llm_models/gguf/'
MODEL_PATH = model_dir + '/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf'

# Пул моделей OpenRouter (от самой мощной к самой слабой)
FREE_MODELS_POOL = [
    "openai/gpt-4o-mini",  # 1. Лидер по логике и структуре
    "google/gemini-2.0-flash-exp:free",  # 2. Высокая скорость и свежие данные
    "meta-llama/llama-3.1-8b-instruct:free",  # 3. Хороший баланс
    "mistralai/mistral-7b-instruct:free",  # 4. Базовый вариант
]

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_batch=1024,
    n_gpu_layers=99,
    n_threads=10,
    verbose=False,
)

# ==============================================================================
# ПРОМПТЫ И КОНСТРУКТОРЫ
# ==============================================================================

system_prompt = (
   "Ты — профессиональный технический консультант по автомобильным жидкостям и маслам. "
   "Отвечай ТОЛЬКО полезной информацией из КОНТЕКСТА. НИКАКИХ заголовков, подзаголовков, слов «Заголовок», «Вёрстка», «Завершение», «Ответ на вопрос», лишних #, **, пустых строк и размышлений.\n\n"


   "КРИТИЧЕСКИЕ ПРАВИЛА:\n"
   "1. Отвечай СРАЗУ с фактов, без вступлений.\n"
   "2. Используй только жирный текст для названий жидкостей, обычный текст для объёмов и спецификаций, • для пунктов.\n"
   "3. Если данных нет — пиши: «Информации по этому узлу в базе нет».\n"
   "4. НИКОГДА не пиши слова: «Заголовок», «Вёрстка», «Завершение», «Ответ на вопрос», «Вот», «Рекомендация».\n"
   "5. Запрещено начинать ответ с # или ##.\n\n"


   "ПРИМЕР ПРАВИЛЬНОГО ОТВЕТА:\n"
   "** для Opel Astra H (L35, L48, L69) 1.4 л, бензин **\n" 
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
   "• Спецификация: ATF DW-1"
)



def promtConstructor(query: str) -> str:
    if 'Mistral' in MODEL_PATH:
        return f"[INST] {system_prompt}\n\n{query} [/INST]"
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def get_max_tokens(param_df_len: int) -> int:
    if param_df_len == 1: return 450
    MAX_TOKENS = 1600
    calculated_tokens = int(170 * param_df_len) + 350
    return calculated_tokens if calculated_tokens < MAX_TOKENS else MAX_TOKENS


# ==============================================================================
# ЛОГИКА КОНТЕКСТА
# ==============================================================================
MAX_len_param_df = 25
LIM_len_param_df = 6





def query_2_Context(client_query: str, param_df, applied_filters, CarParam, oil_and_fluids, PRODUCT_ANCHOR) -> str:
    def clean_context(text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(\s*•\s*){2,}', ' • ', text)
        text = re.sub(r'(?i)внимание.*?(\n\n|\Z)', '\n\n', text, flags=re.DOTALL)
        text = re.sub(r'(?i)примечание.*?(\n\n|\Z)', '\n\n', text, flags=re.DOTALL)
        return text.strip() + '\n\n'

    param_df = param_df[:LIM_len_param_df]
    raw_texts = param_df['A_text'].unique()
    cleaned_texts = [clean_context(text) for text in raw_texts if text.strip()]

    all_A_text = '\n=== ТИП ЖИДКОСТИ === '.join(cleaned_texts) + ' .'

    query = "### КОНТЕКСТ\n<<<ВОПРОС КЛИЕНТА>>> " + client_query
    if not query.endswith("?"): query += "?"
    query += " <<</ВОПРОС КЛИЕНТА>>>"

    if applied_filters:
        lst_param = [f"{param}: {getattr(CarParam, param)}" for param in applied_filters]
        query += "\n<<<ПАРАМЕТРЫ ВОПРОСА>>> " + ", ".join(lst_param) + " <<</ПАРАМЕТРЫ ВОПРОСА>>>"
        query += "\n\n<<<ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ>>>\n=== ТИП ЖИДКОСТИ === " + all_A_text + " <<</ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ>>>"

        ### ### конец <<<ВЫБОР ТОВАРОВ>>> ### ###
        query += ("\n\n<<<ВЫБОР ТОВАРОВ>>>\n"
                  "**КРИТИЧЕСКИ ВАЖНО:**\n: применяй только для моторных масел, трансмиссионных масел и спец-жидкостей.\n"
                  "\n\n'---'\n\n")

        query += ("### АЛГОРИТМ ФИЛЬТРАЦИИ И ПРИОРИТИЗАЦИИ ТОВАРОВ\n\n"
                  "**МАКСИМАЛЬНЫЙ ПРИОРИТЕТ 1 - СПЕЦИФИКАЦИИ И ДОПУСКИ ПРОИЗВОДИТЕЛЯ (OEM) – если допуски OEM указаны выше.\n"
                  "Для трансмиссий особо учитывай: ATF standards, Mercon, Dexron, GL (GL-4/GL-5/GL-5 LS) ZF TE-ML, ALLISON, Voith**\n"
                  "**ПРИОРИТЕТ 2 - ACEA, если спецификация ACEA указана выше**\n"
                  "**ПРИОРИТЕТ 3 - API, если спецификация API указана выше**\n"
                  "**ПРИОРИТЕТ 4 - SAE, если спецификация SAE указана выше**\n"
                  "**МИНИМАЛЬНЫЙ ПРИОРИТЕТ 5 - прочие допуски и спецификации, например: ILSAC, ISO, DOT**\n")

        query += f"\n\n'---'\n\n### ДОСТУПНЫЕ для  ФИЛЬТРАЦИИ И ПРИОРИТИЗАЦИИ ТОВАРЫ\n\n{oil_and_fluids}"


        query += (f"\n\n### Примени АЛГОРИТМ ФИЛЬТРАЦИИ И ПРИОРИТИЗАЦИИ к товарам, указанным выше.\n"
                  f"Для каждого подходящего товара оцени соответствие: ИДЕАЛЬНОЕ / ВЫСОКОЕ / ПРИЕМЛЕМОЕ / НЕ ПОДХОДИТ.\n"
                  f"**КРИТИЧЕСКИ ВАЖНО:**\n"
                  f"1) Если найден хотя бы 1 подходящий товар, начни эту часть ответа (после вывода параметров автомобиля и спецификаций) маркера: {PRODUCT_ANCHOR}\n"
                  f"1.1) Вставь пустую строку (двойной перенос)\n"
                  f"2) Товары, получившие оценку 'НЕ ПОДХОДИТ', не выводи.\n"
                  f"3) КАЖДУЮ строку товара копируй ДОСЛОВНО (verbatim) из списка ДОСТУПНЫЕ ТОВАРЫ — без сокращений и без изменений.\n"
                  f"Используй ВСЁ название из списка ДОСТУПНЫЕ ТОВАРЫ выше, включая артикулы и спецификации\n"
                  f"4) Отсортируй и список по убыванию соответствия требованиям, для каждой товарной категории, выведи не более 3 подходящих товара.\n"
                  f"<<</ВЫБОР ТОВАРОВ>>>")

        ### ### конец <<<ВЫБОР ТОВАРОВ>>> конец ### ###

    query += "\n\n**Задача:** Предоставьте лаконичный, структурированный ответ на запрос клиента:\n" + client_query
    return query


# ==============================================================================
# ЯДРО ГЕНЕРАЦИИ (Cloud + Local)
# ==============================================================================
class BoxResults():
    def __init__(self):
        self.answer = 0
        self.delta_time_LLM = 0
        self.search_type = 0


def aiApi(prompt, max_tokens):
    """Ротация моделей OpenRouter с обработкой ошибок."""
    for model_id in FREE_MODELS_POOL:
        try:
            print(f"--> Trying Cloud API: {model_id}...")
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.2,
                timeout=12
            )
            content = response.choices[0].message.content.strip()
            if content:
                print(f"Ai-Used: Cloud API ({model_id})")
                return content
        except Exception as e:
            print(f"    [SKIP] {model_id} error: {type(e).__name__}")
            continue
    return None


def answerGenerate(query: str, external_ai=False):
    def generate_answer(query_ctx: str, param_df) -> str:
        max_tokens = get_max_tokens(len(param_df))

        # 1. Попытка через облако (Ротация)
        if external_ai:
            answer_cloud = aiApi(query_ctx, max_tokens)
            if answer_cloud: return answer_cloud
            print("!!! All Cloud APIs failed. Falling back to Local Mistral.")

        # 2. Локальный Fallback
        prompt = promtConstructor(query_ctx)
        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2,
            stop=["<|im_end|>", "```", "[/INST]"],
            echo=False,
        )
        return output["choices"][0]["text"].strip()


    BOX = BoxResults()
    CarParam.ssDecoder(query)

    # if not getattr(CarParam, 'global_group') and not getattr(CarParam, 'make'):
    #     BOX.answer = ("Пожалуйста, укажите параметры автомобиля и искомый товар\n"
    #                   "Например: Chery Tiggo 7 pro 2020 масло моторное")
    #     return BOX

    param_dct = {param: getattr(CarParam, param) for param in CarParam.RESULT_ATTRIBUTES}
    param_df, applied_filters = filter_car_params(param_dct)


    if len(param_df) > MAX_len_param_df:
        print(f"\n*** generate_Answer -- > LLM NEW FREE SEARCH")
        result = generate_Answer(query, llm)
        answer = result["answer"]
        BOX.answer = answer
        BOX.search_type = 'LLM_NEW__FREE_SEARCH'

        return BOX

    t1 = U24.tNow()
    oil_and_fluids = ''
    answer = ''
    PRODUCT_ANCHOR = "___Подходящие товары:___"

    if len(param_df) > 0:
        print(f"\n*** query_with_context -- > OIL_CLASSIFIER_PARAMS")
        oil_and_fluids = select_Oil_assortment(param_df, limit_row=8)
        query_with_context = query_2_Context(query, param_df, applied_filters, CarParam, oil_and_fluids, PRODUCT_ANCHOR)
        print(query_with_context)

        answer = generate_answer(query_with_context, param_df)
        if answer:
            BOX.search_type = 'CLASSIFIER_PARAMS'

    if not BOX.search_type:
        print(f"\n*** generate_Answer -- > LLM FREE SEARCH")
        result = generate_Answer(query, llm, k=4)
        answer = result["answer"]
        BOX.search_type = 'LLM_FREE_SEARCH'

    if oil_and_fluids and (not PRODUCT_ANCHOR in answer):
        answer += "\n\n" + oil_and_fluids

    BOX.delta_time_LLM = int((U24.tNow() - t1).total_seconds())
    BOX.answer = answer
    return BOX


# ==============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ И ТЕСТЫ
# ==============================================================================
BOILERPLATE_PREFIXES = [
   'доброго всем здравия',
   'начну с самого начала',
   'подскажите пожалуста',
   'подскажите', 'пожалуйста',
   'нужен совет',
   'спасибо за ответ',
   'вопрос по',
   'несколько вопросов',
   'здравствуйте',
   'скажите',
   'помогите',
   'таки говоря',
   'возник вопрос',
   'и снова',
   'добрый день',
   'доброго времени суток',
   'добрый вечер',
   'доброго дня', 'день добрый', 'доброго времени', 'спасибо'
]

_boiler_re_parts = [rf'(?:^|(?<=[\.\!\?]\s))\s*{re.escape(p)}\s*[,:-]?\s*' for p in BOILERPLATE_PREFIXES]
BOILER_RE = re.compile('|'.join(_boiler_re_parts), re.IGNORECASE | re.UNICODE)


def _strip_boilerplate(s: str) -> str:
    if not isinstance(s, str): return ''
    prev, cur = None, s
    while prev != cur:
        prev = cur
        cur = BOILER_RE.sub('', cur).strip()
    return re.sub(r'\s+', ' ', cur, flags=re.UNICODE).strip()


if __name__ == "__main__":
    # tests = [
    #     'Citroen c-crosser 2011 год, 2,0 литра объем двигателя масло моторное',
    #     'opel astra h 1.6 масло моторное',
    #     'Nissan X-Trail 2012 масло моторное и масло вариантора',
    #     'солярис 2016 моторное масло'
    # ]

    tests = ["FOCUS II - седан - 2.0 - бензин 2012 масло моторное",
             "Подбери масло в ДВС лада гранта 2020, 1.6 л",
              "Changan Lamore подбери трансмиссию",
              "Масло в двигатель g4na",
              "Jetour x50 1.5",
              "Seat Leon масло в МКПП и двигатель",
              "Масло для мотора g4na",
              "Подбери масло для Прадо 2008 года",
              "Land Cruiser Prado 2008",
              "У меня не дизель",
              "Масло трансмиссионное Volvo s60 коробка aisin",

              "Audi a7 масло в коробку dsg",
              "Масло в коробку dsg",
              "volvo s60 2010 масло трансмиссионное",
              "Toyota Camry 2015 какое моторное масло и сколько литров?",
              "Как удалить рекламную наклейку с кузова автомобиля?",
              "Чем почистить карбюратор?",
              "Чем почистить пятно от чая на диване?",
              "Нужны ли антикоррозийные присадки для защиты топливного бака?"
              ]

    T1 = U24.tNow()
    for i, query in enumerate(tests):
        clean_q = _strip_boilerplate(query)
        print(f"\n\n{'=' * 60}{i + 1}/{len(tests)}\nзапрос: {clean_q}")
        # Включаем external_ai=True для использования облака
        BOX = answerGenerate(clean_q, external_ai=True)
        print(f"ответ: {BOX.answer}\n\ndelta_time = {BOX.delta_time_LLM}s")

    print(f"\nСреднее время: {int((U24.tNow() - T1).total_seconds() // len(tests))}s")