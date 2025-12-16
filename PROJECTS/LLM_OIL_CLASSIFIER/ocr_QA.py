#ocr_QA.py = oil_classifier_Question_Answer
from bootstrap import *
from USEFUL_UTILS.ssDecoder_car_bart import ss_2_CarParams
from bootstrap import root_path

CarParam = ss_2_CarParams()

from PROJECTS.LLM_OIL_CLASSIFIER.osr_filter_attributes import filter_car_params
from PROJECTS.LLM_OIL_CLASSIFIER.ocr_QA_universal import generate_Answer
from PROJECTS.LLM_OIL_CLASSIFIER.ocr_select_oil_assortment import select_Oil_assortment


import json
import re
from datetime import datetime
from llama_cpp import Llama

from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Создаём клиент для OpenRouter
client = OpenAI(
    api_key=OPENROUTER_API_KEY,  # ← твой ключ от https://openrouter.ai/keys
    base_url="https://openrouter.ai/api/v1"
)


model_dir = root_path + '/DATA_CATALOGS/llm_models/gguf/'

# MODEL_PATH = model_dir + '//Qwen2.5-Coder-3B-Instruct.Q5_K_M.gguf' #гораздо хуже Mistral
# MODEL_PATH = model_dir + '/DeepSeek-R1-Distill-Qwen-7B.Q4_K_M.gguf' #гораздо хуже Mistral
## Полный отстой: Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf, Phi-3-mini-4k-instruct (Microsoft)

MODEL_PATH = model_dir + '/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf' #наилучший вариант по качеству и скорости




MAX_len_param_df = 25 #в ячейке A_text в среднем 400 символов - кол-во строк больше, значит срез не качественный - общий поиск
LIM_len_param_df = 6 #обрезаем df

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192, #8192
    n_batch=1024,
    n_gpu_layers=99,
    n_threads=10,
    verbose=False,
)

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
    "** для Opel Astra H (L35, L48, L69) 1.4 л, бензин **\n" # "\n"  ** для OPEL ASTRA H 1.4 **
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
    print(f"model = {MODEL_PATH.split('/')[-1]}")
    if 'Mistral' in MODEL_PATH:
        print("promtConstructor = Mistral")
        return f"[INST] {system_prompt}\n\n{query}  [/INST]"

    print("promtConstructor = Universal DeepSeek")
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

#     return f"{system_prompt}\n\n{query}"



def get_max_tokens(param_df_len: int) -> int:
    if param_df_len ==1: return 350
    MAX_TOKENS = 1200
    calculated_tokens =  int(170 * param_df_len) +150
    return calculated_tokens if calculated_tokens < MAX_TOKENS else MAX_TOKENS

def query_2_Context(client_query: str, param_df, applied_filters, CarParam) -> str:
    def clean_context(text: str) -> str:
        # Только безопасные очистки
        text = re.sub(r'\n{3,}', '\n\n', text)  # убираем огромные отступы
        text = re.sub(r'(\s*•\s*){2,}', ' • ', text)  # дубли •
        text = re.sub(r'(?i)внимание.*?(\n\n|\Z)', '\n\n', text, flags=re.DOTALL)
        text = re.sub(r'(?i)примечание.*?(\n\n|\Z)', '\n\n', text, flags=re.DOTALL)
        return text.strip() + '\n\n'



    param_df =  param_df[:LIM_len_param_df]

    raw_texts = param_df['A_text'].unique()
    cleaned_texts = []
    for text in raw_texts:
        cleaned = clean_context(text)  # ← ваша функция
        if cleaned.strip():  # на случай если после очистки пусто
            cleaned_texts.append(cleaned)



    all_A_text = '\n=== ТИП ЖИДКОСТИ === '.join(cleaned_texts) + ' .'

    # all_A_text = f'\n=== ТИП ЖИДКОСТИ === '.join(param_df['A_text'].unique()) +' .'
    print(f"len param_df = {len(param_df)}, len all A_text = {len(all_A_text)}")
    query = "### КОНТЕКСТ\n<<<ВОПРОС КЛИЕНТА>>> " + client_query
    tail = query[-1]
    if tail != "?":
        query += "?"
    query += " <<</ВОПРОС КЛИЕНТА>>>"

    if applied_filters:

        lst_param = []
        for param in applied_filters:
            lst_param.append(f"{param}: {getattr(CarParam, param)}")

        query += "\n<<<ПАРАМЕТРЫ ВОПРОСА>>> " + ", ".join(lst_param) + " <<</ПАРАМЕТРЫ ВОПРОСА>>>"
        query += "\n\n<<<ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ>>>\n=== ТИП ЖИДКОСТИ === " + all_A_text + " <<</ТЕХНИЧЕСКАЯ ИНФОРМАЦИЯ>>>"

    query += "\n\nn**Задача:** Предоставьте лаконичный, структурированный ответ на запрос клиента:\n" + client_query

    return query

class BoxResults():

    def __init__(self):
        self.answer = 0
        self.delta_time_LLM = 0
        self.search_type = 0



def aiApi(prompt, max_tokens):



    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",  # или любая другая модель из списка ниже
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
                temperature=0.2, ###
                stop=["<|im_end|>", "```"],
    )


    return response.choices[0].message.content

def answerGenerate(query: str, external_ai = False):
    def generate_answer(query: str, param_df) -> dict:
        prompt = promtConstructor(query)
        max_tokens=get_max_tokens(len(param_df))

        if external_ai:
            try:
                return aiApi(prompt, max_tokens)

            except Exception as e:
                print(e)


        output = llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.2, ###
            stop=["<|im_end|>", "```"],  # ["<|im_end|>", "```", "\n\n", "\n{"],
            echo=False,
        )

        answer = output["choices"][0]["text"]

        return answer



    BOX = BoxResults() #контейнер для результатов

    CarParam.ssDecoder(query)

    print(f"{query}, {getattr(CarParam, 'make'), getattr(CarParam, 'car'), getattr(CarParam, 'global_group')}")
    if not getattr(CarParam, 'global_group') and not getattr(CarParam, 'make'):
        BOX.answer = ("Пожалуйста, укажите параметры автомобиля и искомый товар\n"
                      "Например: Chery Tiggo 7 pro 2020 масло моторное")

        return BOX


    param_dct = {param:getattr(CarParam, param) for param in CarParam.RESULT_ATTRIBUTES}

    param_df, applied_filters = filter_car_params(param_dct)


    if len(param_df) > MAX_len_param_df:
        print(f"len(param_df) = {len(param_df)}")
        BOX.answer = ("Алгоритм нашёл слишком много вариантов\n"
                      "Попробуйте конкретизировать запрос, подробно опишите ваш автомобиль и необходимые товары\n"
                      "Например: Chery Tiggo 7 pro 2020 масло моторное")

        return BOX



    t1 = U24.tNow()
    oil_and_fluids = ''
    if  len(param_df) > 0:
        print(f"\n*** query_with_context -- > OIL_CLASSIFIER_PARAMS\n\n")
        query_with_context = query_2_Context(query, param_df, applied_filters, CarParam)
        oil_and_fluids = select_Oil_assortment(param_df)
        answer = generate_answer(query_with_context, param_df)
        if answer:
            BOX.search_type = 'CLASSIFIER_PARAMS'



    if not BOX.search_type:
        print(f"\n*** generate_Answer -- > LLM FREE SEARCH\n\n")
        result = generate_Answer(query, llm, k=4)
        answer = result["answer"]
        BOX.search_type = 'LLM_FREE_SEARCH'

    if oil_and_fluids:
        answer += "\n\n" + oil_and_fluids



    BOX.delta_time_LLM = int((U24.tNow() - t1).total_seconds())
    BOX.answer = answer


    return BOX




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

    tests = ['Citroen c-crosser 2011 год, 2,0 литра объем двигателя масло моторное', 'джили монджаро 2023 масло моторное', 'opel astra h 1.6 масло моторное', 'bmw x5 2012 масло моторное',
             'Масло трансмиссионное CVT Chery Tiggo ', 'масло моторное Nisan Maxima 2013', 'Nissan X-Trail 2012 масло моторное и масло вариантора',
             'масло моторное ford Transit 2013', "Audi Q7 2010 масло моторное и трансмисионное", 'жидкость для вариатора Toyota Corolla 2012',
             'солярис 2016 моторное масло']




    """
    

        

    """

    T1 = U24.tNow()
    for i, query in enumerate(tests):
        query = _strip_boilerplate(query)
        print(f"\n\n{'=' * 60}{i}/{len(tests)}\nзапрос: {query}")
        BOX = answerGenerate(query, external_ai=False)
        answer = BOX.answer
        delta_time_LLM = BOX.delta_time_LLM

        print(f"\nзапрос: {query}\nответ: {answer}\n\ndelta_time_LLM = {delta_time_LLM}\n{'=' * 60}")

    print(f"~ среднее время на 1 запрос: {((U24.tNow() - T1).total_seconds())//(i + 1)}")


    """ 

    
    поменять сортировку выдачи - мало моторное вверх
     
       Рейтинг LUKOIL/GAZPROM  
        масло для Опель - Мерседес MERCEDES-BENZ"""






