OLLAMA_MODEL = "qwen3:8b"
FALLBACK_MODEL = "qwen3:4b"

WORDS_PER_ROUND = 5
COARSE_ROUNDS = 6
FINE_ROUNDS = 4
TARGET_HIT_RATE = 0.9
BAND_WIDTH = 500

CONTENT_POS = {"名詞", "動詞-自立", "形容詞", "副詞", "形容動詞", "連体詞"}
SKIP_POS = {"記号", "補助記号"}
FUNCTION_POS = {"助詞", "助動詞"}

STORY_MIN_CHARS = 300
STORY_MAX_CHARS = 600
VOCAB_SAMPLE_SIZE = 300

# Paths relative to the experiment directory (run app from there)
FREQ_LIST_PATH = "data/NLT1.40_freq_list.xlsx"
PARQUET_CACHE_PATH = "data/freq_list.parquet"
PROFILE_PATH = "data/user_profile.json"

CFORM_EXPLANATIONS = {
    "基本形": "dictionary form (base)",
    "連用形": "conjunctive form (used to chain with other verbs/auxiliary)",
    "連用タ接続": "conjunctive form (connecting to た/だ — past tense)",
    "未然形": "irrealis form (used with ない for negation, or れる for passive)",
    "未然ウ接続": "volitional form (let's ~, shall we ~)",
    "仮定形": "conditional form (if ~)",
    "命令形": "imperative form (command)",
    "仮定縮約": "contracted conditional",
    "体言接続": "noun-modifying form",
    "ガル接続": "form connecting to がる (to show signs of ~)",
    "終止形": "sentence-ending form",
}

CTYPE_EXPLANATIONS = {
    "五段・ラ行": "Godan verb, ら-row (う-verb)",
    "五段・カ行イ音便": "Godan verb, か-row with イ euphonic change",
    "五段・サ行": "Godan verb, さ-row",
    "一段": "Ichidan verb (る-verb)",
    "サ変・スル": "する-verb (suru irregular)",
    "カ変・クル": "くる-verb (kuru irregular)",
    "形容詞・アウオ段": "い-adjective",
    "形容詞・イ段": "い-adjective (イ-row)",
    "特殊・ダ": "だ copula",
    "特殊・デス": "です copula (polite)",
    "特殊・タ": "た auxiliary (past tense)",
    "特殊・ナイ": "ない auxiliary (negation)",
}
