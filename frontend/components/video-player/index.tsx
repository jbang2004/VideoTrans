'use client'

import React, { useRef, useState, useEffect } from 'react'
import { cn } from '../../lib/utils'
import { Settings, Play, Pause, Maximize, Minimize } from 'lucide-react'
import { Button } from '../ui/button'
import { VideoControls } from './components/VideoControls'
import { TranslationControls } from './components/TranslationControls'
import { UploadButton } from './components/UploadButton'
import { Sidebar } from './components/Sidebar'
import { useVideoPlayer } from './hooks/useVideoPlayer'
import { useHLSPlayer } from './hooks/useHLSPlayer'
import { useTranslation } from './hooks/useTranslation'
import type { VideoPlayerProps } from './types'

export default function VideoPlayer({ className }: VideoPlayerProps) {
  const { videoRef, state: playerState, controls: playerControls } = useVideoPlayer()
  const { state: translationState, setState: setTranslationState, controls: translationControls } = useTranslation((taskId) => {
    if (taskId) {
      hlsInstance.initHLS(taskId)
    } else {
      hlsInstance.destroyHLS()
    }
  })
  const hlsInstance = useHLSPlayer(videoRef, playerState.isPlaying)
  const [showControls, setShowControls] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const controlsTimeoutRef = useRef<ReturnType<typeof setTimeout>>()
  const containerRef = useRef<HTMLDivElement>(null)

  const handleFileSelect = (file: File) => {
    // 创建本地预览 URL
    const previewUrl = URL.createObjectURL(file)
    playerControls.setLocalVideoUrl(previewUrl)
    setTranslationState(prev => ({
      ...prev,
      selectedFile: file,
      isTranslating: false,
      isProcessing: false,
      taskId: null
    }))
  }

  const toggleFullscreen = () => {
    if (!containerRef.current) return

    if (!isFullscreen) {
      if (containerRef.current.requestFullscreen) {
        containerRef.current.requestFullscreen()
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      }
    }
  }

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(document.fullscreenElement !== null)
    }

    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  const handleMouseMove = (e: React.MouseEvent) => {
    // 检查鼠标是否在底栏区域内
    const target = e.target as HTMLElement
    if (target.closest('.control-bar')) return

    if (!playerState.localVideoUrl && !translationState.taskId) return
    setShowControls(true)
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current)
    }
    controlsTimeoutRef.current = setTimeout(() => {
      setShowControls(false)
    }, 3000)
  }

  const handleMouseEnter = () => {
    setShowControls(true)
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current)
    }
  }

  const handleMouseLeave = () => {
    if (!playerState.localVideoUrl && !translationState.taskId) return
    controlsTimeoutRef.current = setTimeout(() => {
      setShowControls(false)
    }, 3000)
  }

  useEffect(() => {
    return () => {
      if (controlsTimeoutRef.current) {
        clearTimeout(controlsTimeoutRef.current)
      }
    }
  }, [])

  return (
    <div className="flex h-screen bg-gradient-to-br from-neutral-950 to-black text-white overflow-hidden">
      {/* Sidebar */}
      {!isFullscreen && <Sidebar />}

      {/* Main Content - Video Player */}
      <div 
        ref={containerRef}
        className={cn(
          "flex-1 relative min-w-0",
          isFullscreen && "bg-black"
        )}
        onMouseMove={handleMouseMove}
      >
        <div className="absolute inset-0 flex items-center justify-center bg-black">
          <div className="relative w-full h-full max-h-screen">
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-contain"
              playsInline
            />

            {/* Overlay Controls */}
            <div className="absolute inset-0 flex items-center justify-center">
              {(!playerState.localVideoUrl && !translationState.taskId) && (
                <div className="flex items-center justify-center gap-8">
                  <UploadButton
                    onFileSelect={handleFileSelect}
                    className="h-20 w-20 rounded-full bg-white/20 hover:bg-white/30 backdrop-blur-md active:scale-95 transition-all shadow-xl"
                  />
                </div>
              )}
            </div>

            {/* Bottom Control Bar */}
            <div 
              className={cn(
                "control-bar absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent pt-20 pb-6 px-4 transition-opacity duration-300",
                showControls ? "opacity-100" : "opacity-0"
              )}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
            >
              <div className="mx-auto flex flex-col items-center justify-between bg-black/60 backdrop-blur-sm border border-white/10 rounded-xl px-4 py-2 gap-2 w-full max-w-3xl shadow-2xl">
                <VideoControls 
                  state={playerState}
                  controls={playerControls}
                />
                <div className="flex items-center justify-between w-full">
                  <div className="flex items-center gap-4">
                    <UploadButton
                      onFileSelect={handleFileSelect}
                      className="hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white"
                    />
                    <Button 
                      size="icon" 
                      variant="ghost" 
                      className="hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white"
                      onClick={() => {
                        if (playerState.isPlaying) {
                          playerControls.pause()
                        } else {
                          playerControls.play()
                        }
                      }}
                    >
                      {playerState.isPlaying ? (
                        <Pause className="h-4 w-4" />
                      ) : (
                        <Play className="h-4 w-4 ml-0.5" />
                      )}
                    </Button>
                  </div>
                  <div className="flex items-center gap-4">
                    <TranslationControls
                      state={translationState}
                      controls={translationControls}
                    />
                    <Button 
                      size="icon" 
                      variant="ghost" 
                      className="hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white"
                      onClick={toggleFullscreen}
                    >
                      {isFullscreen ? (
                        <Minimize className="h-4 w-4" />
                      ) : (
                        <Maximize className="h-4 w-4" />
                      )}
                    </Button>
                    <Button 
                      size="icon" 
                      variant="ghost" 
                      className="hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white"
                    >
                      <Settings className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
} 