// ===============================================
// frontend/components/video-player/components/TranslationControls.tsx
// ===============================================
import React, { useEffect } from "react"
import { Button } from "../../ui/button"
import { Popover, PopoverTrigger, PopoverContent } from "../../ui/popover"
import { ChevronUp, Languages, Subtitles, ArrowRight, Upload, Loader2 } from "lucide-react"
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
    subtitleWanted,
    // 新增状态
    isUploaded,
    isPreprocessing,
    isPreprocessed
  } = state

  const {
    startTranslation,
    stopTranslation,
    setLanguage,
    toggleSubtitleWanted,
    // 新增方法
    uploadVideo,
    preprocessVideo
  } = controls

  // 当文件被选择后自动上传 - 和index.tsx中的逻辑一起确保上传流程
  useEffect(() => {
    if (selectedFile && !isUploaded && !isProcessing) {
      uploadVideo();
    }
  }, [selectedFile, isUploaded, isProcessing]);

  // 语言选择 handle
  const handleLanguageSelect = (language: string) => {
    setLanguage(language)
  }

  // 根据当前状态决定按钮行为和文字
  const getButtonProps = () => {
    // 如果已经完成，显示下载按钮
    if (isCompleted) {
      return {
        text: "下载",
        onClick: () => window.open(`${API_BASE_URL}/download/${taskId}`, "_blank"),
        disabled: false,
        icon: null
      }
    }
    
    // 如果正在翻译中，显示翻译中状态
    if (isTranslating) {
      return {
        text: "翻译中",
        onClick: () => {},
        disabled: true,
        icon: <Loader2 className="h-3 w-3 mr-1 animate-spin" />
      }
    }
    
    // 如果正在预处理中，显示预处理中状态
    if (isPreprocessing) {
      return {
        text: "预处理中",
        onClick: () => {},
        disabled: true,
        icon: <Loader2 className="h-3 w-3 mr-1 animate-spin" />
      }
    }
    
    // 如果正在上传中，显示上传中状态
    if (isProcessing && !isPreprocessing && !isTranslating) {
      return {
        text: "上传中",
        onClick: () => {},
        disabled: true,
        icon: <Loader2 className="h-3 w-3 mr-1 animate-spin" />
      }
    }
    
    // 如果预处理完成，显示翻译按钮
    if (isPreprocessed) {
      return {
        text: "翻译",
        onClick: startTranslation,
        disabled: false,
        icon: <ArrowRight className="h-3 w-3 mr-1" />
      }
    }
    
    // 如果已上传但未预处理，显示预处理按钮
    if (isUploaded) {
      return {
        text: "预处理",
        onClick: preprocessVideo,
        disabled: false,
        icon: <ArrowRight className="h-3 w-3 mr-1" />
      }
    }
    
    // 如果有文件但还未上传完成，显示正在上传
    if (selectedFile && !isUploaded) {
      return {
        text: "上传中",
        onClick: () => {},
        disabled: true, 
        icon: <Loader2 className="h-3 w-3 mr-1 animate-spin" />
      }
    }
    
    // 默认情况，没有文件
    return {
      text: "处理",
      onClick: () => {},
      disabled: true,
      icon: null
    }
  }

  const buttonProps = getButtonProps()

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
        <PopoverContent className="w-28 bg-neutral-800/90 backdrop-blur-lg border-white/10 rounded-xl shadow-2xl p-1.5" side="top" sideOffset={5}>
          <div className="space-y-0.5">
            {LANGUAGES.map((language) => (
              <Button
                key={language.value}
                size="sm"
                variant="ghost"
                className="w-full justify-start text-xs px-2 py-1 h-7 text-white/80 hover:text-white hover:bg-white/10 rounded-lg"
                onClick={(e) => {
                  e.stopPropagation(); // 防止点击穿透
                  handleLanguageSelect(language.label);
                }}
              >
                {language.label}
              </Button>
            ))}
          </div>
        </PopoverContent>
      </Popover>

      {/* 字幕开关按钮 */}
      <Button
        size="sm"
        variant="ghost"
        className="h-7 px-2.5 rounded-full hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white text-xs"
        // 一旦开始预处理或翻译或已完成，就不可再改
        disabled={isPreprocessing || isTranslating || isCompleted || isProcessing || isPreprocessed}
        onClick={() => toggleSubtitleWanted()}
      >
        <Subtitles className="h-3 w-3 mr-1 opacity-70" />
        {subtitleWanted ? '开' : '关'}
      </Button>

      {/* 主操作按钮：上传/预处理/翻译/下载 */}
      <Button
        size="sm"
        variant="ghost"
        className={cn(
          "h-7 px-2.5 rounded-full hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white text-xs",
          (selectedFile || isUploaded || isPreprocessed || isTranslating || isCompleted) && "bg-white/10"
        )}
        disabled={buttonProps.disabled}
        onClick={buttonProps.onClick}
      >
        {buttonProps.icon}
        {buttonProps.text}
      </Button>
    </div>
  )
}