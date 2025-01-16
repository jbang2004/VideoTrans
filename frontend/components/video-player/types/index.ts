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

export interface TranslationState {
  isTranslating: boolean
  isProcessing: boolean
  selectedLanguage: string
  taskId: string | null
  selectedFile: File | null
}

export interface TranslationControls {
  startTranslation: () => Promise<void>
  stopTranslation: () => void
  setLanguage: (language: string) => void
}

export interface HLSInstance {
  initHLS: (taskId: string) => void
  destroyHLS: () => void
}

export interface VideoPlayerProps {
  className?: string
} 