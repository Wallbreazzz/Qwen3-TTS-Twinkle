"""通用多码本 TTS Preprocessor 基类（模型无关）。

把 manifest 行打包成 Twinkle `Trajectory`：text 放 `messages`，audio_codes / ref_audio /
language 放 `user_data`（PyArrow-stable，用 `pack_value` 编码）。下游自定义 Template
（如 `tts.qwen3.template.Qwen3TTSTemplate`）从 `user_data` 取出 audio_codes 构建具体
InputFeature。

manifest 字段名用类属性约定，子类可覆盖适配不同数据集（Qwen3 用默认即可）。
"""

from typing import Any, Dict

from twinkle.data_format import Message, Trajectory, pack_value
from twinkle.preprocessor import Preprocessor


class MultiCodebookPreprocessor(Preprocessor):
    """manifest 行 → Trajectory（多码本 audio_codes 走 user_data）。

    子类一般只需覆盖字段名类属性（或直接用默认），复杂定制可覆盖 `preprocess`。
    """

    # manifest 字段名约定（子类可覆盖）
    TEXT_FIELD = "text"
    AUDIO_CODES_FIELD = "audio_codes"          # [[...], shape C×N]
    REF_AUDIO_FIELD = "ref_audio"              # 参考音频路径（或预编码 codes，视模型而定）
    LANGUAGE_FIELD = "language"
    DEFAULT_LANGUAGE = "Auto"

    def __call__(self, rows: Dict[str, list]) -> Dict[str, list]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(r) for r in rows]
        return self.map_row_to_col(rows)

    def preprocess(self, row) -> Trajectory:
        text = row[self.TEXT_FIELD]
        audio_codes = row[self.AUDIO_CODES_FIELD]
        ref_audio = row.get(self.REF_AUDIO_FIELD)
        language = row.get(self.LANGUAGE_FIELD, self.DEFAULT_LANGUAGE)
        return Trajectory(
            messages=[Message(role="user", content=text)],
            user_data=[
                ("audio_codes", pack_value(audio_codes)),
                ("ref_audio", pack_value(ref_audio)),
                ("language", pack_value(language)),
            ],
        )
