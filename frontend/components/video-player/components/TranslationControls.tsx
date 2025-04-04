// ===============================================
// frontend/components/video-player/components/TranslationControls.tsx
// ===============================================
import React from "react"
import { Button } from "../../ui/button"
import { Popover, PopoverTrigger, PopoverContent } from "../../ui/popover"
import { ChevronUp, Languages, Subtitles, ArrowRight } from "lucide-react"
import { cn } from "../../../lib/utils"
import { LANGUAGES, API_BASE_URL } from "../utils/format"
import type { TranslationState, TranslationControls } from "../types"

interface TranslationControlsProps {
  state: TranslationState
  controls: TranslationControls
}

export function TranslationControls({ state, controls }: TranslationControlsProps) {
  const {
    isTranslating,
    isProcessing,
    isCompleted,
    selectedLanguage,
    selectedFile,
    taskId,
    // ============ (新增) ============
    subtitleWanted,
  } = state

  const {
    startTranslation,
    stopTranslation,
    setLanguage,
    // ============ (新增) ============
    toggleSubtitleWanted,
  } = controls

  // 语言选择 handle
  const handleLanguageSelect = (language: string) => {
    setLanguage(language)
  }

  // 主按钮
  let buttonText = "翻译"
  let buttonIcon = <ArrowRight className="h-3 w-3" />
  
  if (isCompleted) {
    buttonText = "下载"
  } else if (isTranslating) {
    buttonText = "处理中"
  }

  const handleMainButtonClick = async () => {
    if (isCompleted && taskId) {
      window.open(`${API_BASE_URL}/download/${taskId}`, "_blank")
      return
    }
    if (isTranslating) {
      stopTranslation()
      return
    }
    await startTranslation()
  }

  return (
    <div className="flex items-center gap-2">
      {/* 语言选择下拉 */}
      <Popover>
        <PopoverTrigger asChild>
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2.5 rounded-full hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white text-xs"
          >
            <Languages className="h-3 w-3 mr-1 opacity-70" />
            {selectedLanguage}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-28 bg-neutral-800/90 backdrop-blur-lg border-white/10 rounded-xl shadow-2xl p-1.5">
          <div className="space-y-0.5">
            {LANGUAGES.map((language) => (
              <Button
                key={language.value}
                size="sm"
                variant="ghost"
                className="w-full justify-start text-xs px-2 py-1 h-7 text-white/80 hover:text-white hover:bg-white/10 rounded-lg"
                onClick={() => handleLanguageSelect(language.label)}
              >
                {language.label}
              </Button>
            ))}
          </div>
        </PopoverContent>
      </Popover>

      {/* ============== (新增) 字幕开关按钮 ============== */}
      <Button
        size="sm"
        variant="ghost"
        className="h-7 px-2.5 rounded-full hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white text-xs"
        // 一旦开始翻译 or 已完成，就不可再改
        disabled={isTranslating || isCompleted || isProcessing}
        onClick={() => toggleSubtitleWanted()}
      >
        <Subtitles className="h-3 w-3 mr-1 opacity-70" />
        {subtitleWanted ? '开' : '关'}
      </Button>

      {/* 单个主按钮 => 开始翻译 / 翻译中 / 下载 */}
      <Button
        size="sm"
        variant="ghost"
        className={cn(
          "h-7 px-2.5 rounded-full hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white text-xs",
          (selectedFile || isTranslating || isCompleted) && "bg-white/10"
        )}
        // 若正在处理且没到完成, 也可禁用
        disabled={isProcessing && !isCompleted}
        onClick={handleMainButtonClick}
      >
        {isTranslating && (
          <span className="mr-1 h-3 w-3 inline-block animate-pulse rounded-full bg-emerald-400"></span>
        )}
        {buttonText}
      </Button>
    </div>
  )
}