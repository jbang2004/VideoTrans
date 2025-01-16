import React from 'react'
import { Button } from '../../ui/button'
import { Popover, PopoverTrigger, PopoverContent } from '../../ui/popover'
import { ChevronUp } from 'lucide-react'
import { cn } from '../../../lib/utils'
import { LANGUAGES } from '../utils/format'
import type { TranslationState, TranslationControls } from '../types'

interface TranslationControlsProps {
  state: TranslationState
  controls: TranslationControls
}

export function TranslationControls({ state, controls }: TranslationControlsProps) {
  const { isTranslating, isProcessing, selectedLanguage, selectedFile } = state
  const { startTranslation, stopTranslation, setLanguage } = controls

  const handleLanguageSelect = (language: string) => {
    setLanguage(language)
  }

  const toggleTranslation = () => {
    if (isTranslating) {
      stopTranslation()
    } else {
      startTranslation()
    }
  }

  return (
    <div className="flex items-center gap-4">
      <Popover>
        <PopoverTrigger asChild>
          <Button 
            variant="ghost" 
            className="text-sm hover:bg-white/10 active:scale-95 transition-transform gap-2 text-white/70 hover:text-white"
          >
            {selectedLanguage}
            <ChevronUp className="h-4 w-4" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-32 bg-black/60 backdrop-blur-xl border-white/20 rounded-xl shadow-2xl">
          <div className="space-y-1">
            {LANGUAGES.map((language) => (
              <Button
                key={language.value}
                variant="ghost"
                className={cn(
                  "w-full justify-start text-white/70 hover:text-white hover:bg-white/10",
                  selectedLanguage === language.label && "bg-white/10"
                )}
                onClick={() => handleLanguageSelect(language.label)}
              >
                {language.label}
              </Button>
            ))}
          </div>
        </PopoverContent>
      </Popover>
      <Button 
        variant="ghost" 
        className={cn(
          "text-sm hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white",
          (selectedFile || isTranslating) && "bg-white/10"
        )}
        onClick={toggleTranslation}
        disabled={!selectedFile || isProcessing}
      >
        {isProcessing ? "处理中..." : (isTranslating ? "停止翻译" : "开始翻译")}
      </Button>
    </div>
  )
} 