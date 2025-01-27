// ==================================
// frontend/components/video-player/types/index.ts
// ==================================
export interface PlayerState {
  isPlaying: boolean
  currentTime: number
  duration: number
  volume: number
  localVideoUrl: string | null
}

export interface PlayerControls {
  play: () => void
  pause: () => void
  seek: (time: number) => void
  setVolume: (volume: number) => void
  setLocalVideoUrl: (url: string | null) => void
}

// ====================== (新增字段) ======================
export interface TranslationState {
  isTranslating: boolean
  isProcessing: boolean
  selectedLanguage: string
  taskId: string | null
  selectedFile: File | null
  isCompleted: boolean

  // 是否“想要烧制字幕”
  subtitleWanted: boolean
}

// ====================== (新增方法) ======================
export interface TranslationControls {
  startTranslation: () => Promise<void>
  stopTranslation: () => void
  setLanguage: (language: string) => void

  // 切换“字幕Wanted”的布尔值
  toggleSubtitleWanted: () => void
}

export interface HLSInstance {
  initHLS: (taskId: string) => void
  destroyHLS: () => void
}

export interface VideoPlayerProps {
  className?: string
}
