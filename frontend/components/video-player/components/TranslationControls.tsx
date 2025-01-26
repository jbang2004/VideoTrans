import React from "react"
import { Button } from "../../ui/button"
import { Popover, PopoverTrigger, PopoverContent } from "../../ui/popover"
import { ChevronUp } from "lucide-react"
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
    taskId
  } = state

  const {
    startTranslation,
    stopTranslation,
    setLanguage
  } = controls

  // 1) 根据当前状态确定按钮文字
  let buttonText = "开始翻译"
  if (isCompleted) {
    buttonText = "下载"
  } else if (isTranslating) {
    buttonText = "翻译中"
  }

  // 2) 主按钮点击逻辑
  const handleMainButtonClick = async () => {
    // 若已完成翻译 => 点击则下载
    if (isCompleted && taskId) {
      window.open(`${API_BASE_URL}/download/${taskId}`, "_blank")
      return
    }
    // 若正在翻译 => 点击则停止
    if (isTranslating) {
      stopTranslation()
      return
    }
    // 否则 => 开始翻译
    await startTranslation()
  }

  // 3) 切换语言
  const handleLanguageSelect = (language: string) => {
    setLanguage(language)
  }

  return (
    <div className="flex items-center gap-4">
      {/* 语言选择下拉 */}
      <Popover>
        <PopoverTrigger asChild>
          <Button
            variant="ghost"
            className="text-sm hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white"
          >
            {selectedLanguage}
            <ChevronUp className="h-4 w-4 ml-1" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-32 bg-black/60 backdrop-blur-xl border-white/20 rounded-xl shadow-2xl">
          <div className="space-y-1">
            {LANGUAGES.map((language) => (
              <Button
                key={language.value}
                variant="ghost"
                className="w-full justify-start text-white/80 hover:text-white hover:bg-white/10"
                onClick={() => handleLanguageSelect(language.label)}
              >
                {language.label}
              </Button>
            ))}
          </div>
        </PopoverContent>
      </Popover>

      {/* 单个主按钮 => 开始翻译 / 翻译中 / 下载 */}
      <Button
        variant="ghost"
        className={cn(
          "text-sm hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white",
          (selectedFile || isTranslating || isCompleted) && "bg-white/10"
        )}
        // 若正在处理且没到完成, 可以禁用以防误点; 根据需要可灵活调整
        disabled={isProcessing && !isCompleted}
        onClick={handleMainButtonClick}
      >
        {buttonText}
      </Button>
    </div>
  )
}
