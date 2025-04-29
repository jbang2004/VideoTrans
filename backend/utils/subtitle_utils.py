import pysubs2
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def generate_subtitles_for_segment(
    sentences: List[Any],
    start_time_ms: float,
    output_sub_path: str,
    target_language: str = "en"
):
    """
    使用 pysubs2 生成 ASS 字幕文件.
    1. 遍历每条 Sentence, 将其文本拆分成若干块(避免过长字幕).
    2. 向 pysubs2 中写入事件, 并设置"类YouTube"的默认样式.
    3. 最后保存为 .ass 文件.

    Args:
        sentences: 本段的句子列表
        start_time_ms: 当前片段的起始时间（毫秒）
        output_sub_path: 存放字幕的 .ass 路径
        target_language: 用来确定拆分逻辑(中文/英文/日文/韩文)
    """
    subs = pysubs2.SSAFile()

    for s in sentences:
        # 计算相对时间
        start_local = s.adjusted_start - start_time_ms

        sub_text = (s.trans_text or s.raw_text or "").strip()
        if not sub_text:
            continue

        # 直接使用adjusted_duration作为duration_ms
        if s.speed <= 0.0001:  # 使用一个很小的阈值而不是直接判断=0
            # 如果speed接近0，直接使用adjusted_duration
            duration_ms = s.adjusted_duration
            logger.warning(f"检测到speed接近0，直接使用adjusted_duration({s.adjusted_duration}ms): {s.trans_text}")
        else:
            # 正常情况下使用计算公式
            duration_ms = s.duration / s.speed
        if duration_ms <= 0:
            continue

        # 如果 Sentence 本身带 lang, 就优先使用 s.lang, 否则用 target_language
        lang = target_language or "en"

        # 拆分长句子 -> 多段 sequential
        blocks = split_long_text_to_sub_blocks(
            text=sub_text,
            start_ms=start_local,
            duration_ms=duration_ms,
            lang=lang
        )

        for block in blocks:
            evt = pysubs2.SSAEvent(
                start=int(block["start"]),
                end=int(block["end"]),
                text=block["text"]
            )
            subs.append(evt)

    # 设置"类YouTube"的默认样式
    # 若 "Default" 不存在则创建
    style = subs.styles.get("Default", pysubs2.SSAStyle())

    style.fontname = "Arial"             # 常见无衬线
    style.fontsize = 22
    style.bold = True
    style.italic = False
    style.underline = False

    # 颜色 (R, G, B, A=0 => 不透明)
    # 文字: 白色,  描边/背景: 黑色
    style.primarycolor = pysubs2.Color(255, 255, 255, 0)
    style.outlinecolor = pysubs2.Color(0, 0, 0, 100)  # 半透明黑
    style.borderstyle = 3  # 3 => 有背景块
    # style.outline = 4      # 背景矩形厚度
    style.shadow = 0
    style.alignment = pysubs2.Alignment.BOTTOM_CENTER
    style.marginv = 20    # 离底部像素
    # style.marginl = 30      # 左边距，根据需要调整
    # style.marginr = 30      # 右边距，根据需要调整

    # 更新回 default
    subs.styles["Default"] = style

    # 写入文件
    subs.save(output_sub_path, format="ass")
    logger.debug(f"generate_subtitles_for_segment: 已写入字幕 => {output_sub_path}")

def split_long_text_to_sub_blocks(
    text: str,
    start_ms: float,
    duration_ms: float,
    lang: str = "en"
) -> List[Dict[str, Any]]:
    """
    将文本在 [start_ms, start_ms+duration_ms] 区间内拆分成多块 sequential 字幕，
    并按照每块的字符数在总时长内做比例分配。

    Args:
        text: 要拆分的文本
        start_ms: 开始时间（毫秒）
        duration_ms: 持续时间（毫秒）
        lang: 语言代码，"zh"/"ja"/"ko"/"en"，若无匹配则默认英文
        
    Returns:
        字幕块列表，每个块包含开始时间、结束时间和文本
    """
    # 1) 确定每行最大字符数
    recommended_max_chars = {
        "zh": 20,
        "ja": 20,
        "ko": 20,
        "en": 40
    }
    if lang not in recommended_max_chars:
        lang = "en"
    max_chars = recommended_max_chars[lang]

    if len(text) <= max_chars:
        # 不需要拆分
        return [{
            "start": start_ms,
            "end":   start_ms + duration_ms,
            "text":  text
        }]

    # 2) 根据语言拆分成若干行
    chunks = chunk_text_by_language(text, lang, max_chars)

    # 3) 根据各行的字符数在总时长内进行时间分配
    sub_blocks = []
    total_chars = sum(len(c) for c in chunks)
    current_start = start_ms

    for c in chunks:
        chunk_len = len(c)
        chunk_dur = duration_ms * (chunk_len / total_chars) if total_chars > 0 else 0
        block_start = current_start
        block_end   = current_start + chunk_dur

        sub_blocks.append({
            "start": block_start,
            "end":   block_end,
            "text":  c
        })
        current_start += chunk_dur

    # 修正最后一块结束
    if sub_blocks:
        sub_blocks[-1]["end"] = start_ms + duration_ms
    else:
        # 理论上不会发生
        sub_blocks.append({
            "start": start_ms,
            "end":   start_ms + duration_ms,
            "text":  text
        })

    return sub_blocks

def chunk_text_by_language(text: str, lang: str, max_chars: int) -> List[str]:
    """
    根据语言做拆分:
     - 英文: 按单词拆, 避免截断单词
     - 中/日/韩: 按字符拆, 尝试在标点附近断行
     
    Args:
        text: 要拆分的文本
        lang: 语言代码
        max_chars: 每行最大字符数
        
    Returns:
        拆分后的文本块列表
    """
    cjk_puncts = set("，,。.!！？?；;：:、…~— ")
    eng_puncts = set(".,!?;: ")

    if lang == "en":
        return chunk_english_text(text, max_chars, eng_puncts)
    else:
        return chunk_cjk_text(text, max_chars, cjk_puncts)

def chunk_english_text(text: str, max_chars: int, puncts: set) -> List[str]:
    """
    英文文本拆分，按单词边界拆分
    
    Args:
        text: 英文文本
        max_chars: 每行最大字符数
        puncts: 标点符号集合
        
    Returns:
        拆分后的文本块列表
    """
    words = text.split()
    chunks = []
    current_line = []

    for w in words:
        # 计算本行加上下一个单词后长多少
        line_len = sum(len(x) for x in current_line) + len(current_line)  # 单词总长 + 空格数
        if line_len + len(w) > max_chars:
            if current_line:
                chunks.append(" ".join(current_line))
                current_line = []
        current_line.append(w)

    # 收尾
    if current_line:
        chunks.append(" ".join(current_line))

    return chunks

def chunk_cjk_text(text: str, max_chars: int, puncts: set) -> List[str]:
    """
    中日韩文本拆分，尝试在标点处断行
    
    Args:
        text: 中日韩文本
        max_chars: 每行最大字符数
        puncts: 标点符号集合
        
    Returns:
        拆分后的文本块列表
    """
    chunks = []
    total_length = len(text)
    start_idx = 0

    while start_idx < total_length:
        # 基础结束位置
        end_idx = start_idx + max_chars
        
        # 如果下一个字符是标点，则包含到当前块
        if end_idx < total_length and text[end_idx] in puncts:
            end_idx += 1  # 包含标点符号

        # 确保不越界
        end_idx = min(end_idx, total_length)
        
        # 截取当前块
        chunk = text[start_idx:end_idx]
        chunks.append(chunk)
        
        # 移动起始位置
        start_idx = end_idx

    return chunks 