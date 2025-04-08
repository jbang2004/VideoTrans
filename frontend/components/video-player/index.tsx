// ===============================================
// frontend/components/video-player/index.tsx
// ===============================================
'use client'

import React, { useRef, useState, useEffect } from 'react'
import { cn } from '../../lib/utils'
import { Settings, Play, Pause, Maximize, Minimize, Volume2 } from 'lucide-react'
import { Button } from '../ui/button'
import { TranslationControls } from './components/TranslationControls'
import { UploadButton } from './components/UploadButton'
import { useVideoPlayer } from './hooks/useVideoPlayer'
import { useHLSPlayer } from './hooks/useHLSPlayer'
import { useTranslation } from './hooks/useTranslation'
import { Slider } from '../ui/slider'
import { formatTime } from './utils/format'

interface VideoPlayerProps {
  initialFile?: File
}

export default function VideoPlayer({ initialFile }: VideoPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const { videoRef, state: playerState, controls: playerControls } = useVideoPlayer()
  const [showControls, setShowControls] = useState(true)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [showVolumeSlider, setShowVolumeSlider] = useState(false)

  // ========== 新增: 由useTranslation()管理字幕Wanted等 ==========
  const { state: translationState, setState: setTranslationState, controls: translationControls } = useTranslation((taskId) => {
    if (taskId) {
      console.log(`准备初始化HLS播放器，任务ID: ${taskId}`)
      hlsInstance.initHLS(taskId)
    } else {
      console.log('销毁HLS播放器')
      hlsInstance.destroyHLS()
    }
  })

  const hlsInstance = useHLSPlayer(videoRef, playerState.isPlaying)
  const controlsTimeoutRef = useRef<ReturnType<typeof setTimeout>>()

  const handleFileSelect = (file: File) => {
    const previewUrl = URL.createObjectURL(file)
    playerControls.setLocalVideoUrl(previewUrl)
    setTranslationState(prev => ({
      ...prev,
      selectedFile: file,
      isTranslating: false,
      isProcessing: false,
      taskId: null,
      isCompleted: false,
      hlsReady: false
    }))
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!playerState.localVideoUrl && !translationState.taskId) return
    setShowControls(true)
    if (controlsTimeoutRef.current) {
      clearTimeout(controlsTimeoutRef.current)
    }
    controlsTimeoutRef.current = setTimeout(() => {
      setShowControls(false)
      setShowVolumeSlider(false) // 隐藏音量滑块当控制栏消失时
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
      setShowVolumeSlider(false) // 隐藏音量滑块当控制栏消失时
    }, 3000)
  }

  // 全屏切换
  const toggleFullscreen = () => {
    if (!containerRef.current) return
    
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().then(() => {
        setIsFullscreen(true)
      }).catch(err => {
        console.error(`全屏请求失败: ${err.message}`)
      })
    } else {
      document.exitFullscreen().then(() => {
        setIsFullscreen(false)
      }).catch(err => {
        console.error(`退出全屏失败: ${err.message}`)
      })
    }
  }

  // 切换音量滑块显示状态
  const toggleVolumeSlider = (e: React.MouseEvent) => {
    e.stopPropagation() // 防止事件冒泡
    setShowVolumeSlider(!showVolumeSlider)
  }

  // 自动清理控制器超时
  useEffect(() => {
    return () => {
      if (controlsTimeoutRef.current) {
        clearTimeout(controlsTimeoutRef.current)
      }
    }
  }, [])

  // 监听全屏变化
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => {
      document.removeEventListener('fullscreenchange', handleFullscreenChange)
    }
  }, [])

  // 点击其他地方时隐藏音量滑块
  useEffect(() => {
    const handleClickOutside = () => {
      setShowVolumeSlider(false)
    }
    
    if (showVolumeSlider) {
      document.addEventListener('click', handleClickOutside)
    }
    
    return () => {
      document.removeEventListener('click', handleClickOutside)
    }
  }, [showVolumeSlider])

  // 处理初始文件
  useEffect(() => {
    if (initialFile) {
      handleFileSelect(initialFile)
    }
  }, [initialFile])

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-neutral-950 to-black text-white overflow-hidden">
      {/* 主视频区域 */}
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
            {/* Video元素 */}
            <video
              ref={videoRef}
              className="absolute inset-0 w-full h-full object-contain"
              playsInline
            />

            {/* 加载中状态提示 - 当任务已创建但HLS尚未就绪时显示 */}
            {translationState.isTranslating && translationState.taskId && !translationState.hlsReady && (
              <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-70 z-10">
                <div className="text-center">
                  <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent align-[-0.125em] text-primary motion-reduce:animate-[spin_1.5s_linear_infinite]"></div>
                  <p className="mt-4 text-white">正在准备视频流，请稍候...</p>
                </div>
              </div>
            )}

            {/* Overlay: 当没选视频时，中心显示Upload */}
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
            
            {/* 底部控制栏 */}
            <div 
              className={cn(
                "control-bar absolute bottom-0 left-0 right-0 pb-5 px-4 transition-opacity duration-300 z-20",
                showControls ? "opacity-100" : "opacity-0"
              )}
              onMouseEnter={handleMouseEnter}
              onMouseLeave={handleMouseLeave}
            >
              <div className="mx-auto flex items-center justify-between bg-neutral-800/60 backdrop-blur-lg border border-white/10 rounded-full px-4 py-2 gap-3 w-full max-w-3xl shadow-2xl">
                {/* 左侧控件：上传和播放/暂停 */}
                <div className="flex items-center gap-2">
                  <UploadButton
                    onFileSelect={handleFileSelect}
                    className="hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white"
                  />
                  <Button 
                    size="icon" 
                    variant="ghost" 
                    className="hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white h-7 w-7"
                    onClick={() => {
                      if (playerState.isPlaying) {
                        playerControls.pause()
                      } else {
                        playerControls.play()
                      }
                    }}
                  >
                    {playerState.isPlaying ? (
                      <Pause className="h-3.5 w-3.5" />
                    ) : (
                      <Play className="h-3.5 w-3.5 ml-0.5" />
                    )}
                  </Button>
                </div>

                {/* 中间：进度条 */}
                <div className="flex-1 flex items-center gap-1.5 max-w-lg mx-1.5">
                  <span className="text-xs text-white/70 min-w-[36px]">{formatTime(playerState.currentTime)}</span>
                  <div className="relative w-full h-1 group">
                    <div className="absolute inset-0 bg-white/20 rounded-full" />
                    <div 
                      className="absolute inset-y-0 left-0 bg-white/80 rounded-full transition-all"
                      style={{ width: `${(playerState.currentTime / playerState.duration) * 100}%` }}
                    />
                    <Slider
                      value={[playerState.currentTime]}
                      max={playerState.duration}
                      step={0.1}
                      className="absolute inset-0 appearance-none bg-transparent [&>span]:opacity-0 group-hover:[&>span]:opacity-100 [&>span]:transition-opacity [&>span]:duration-200"
                      onValueChange={(value) => playerControls.seek(value[0])}
                    />
                  </div>
                  <span className="text-xs text-white/70 min-w-[36px]">{formatTime(playerState.duration)}</span>
                </div>

                {/* 右侧控件：翻译、音量、全屏 */}
                <div className="flex items-center gap-2">
                  {/* 音量控制 */}
                  <div className="relative flex items-center">
                    <Button 
                      size="icon" 
                      variant="ghost" 
                      className="hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white h-7 w-7"
                      onClick={toggleVolumeSlider}
                    >
                      <Volume2 className="h-3 w-3" />
                    </Button>
                    {showVolumeSlider && (
                      <div className="absolute left-1/2 -translate-x-1/2 bottom-full mb-2 bg-neutral-800/90 backdrop-blur-lg rounded-full px-2 py-4 shadow-xl">
                        <div className="relative h-[80px] w-1 group mx-auto">
                          <div className="absolute inset-0 bg-white/20 rounded-full" />
                          <div 
                            className="absolute inset-x-0 bottom-0 bg-white/80 rounded-full transition-all"
                            style={{ height: `${playerState.volume * 100}%` }}
                          />
                          <Slider
                            orientation="vertical"
                            value={[playerState.volume]}
                            max={1}
                            step={0.01}
                            className="absolute inset-0 appearance-none bg-transparent [&>span]:opacity-0"
                            onValueChange={(value) => playerControls.setVolume(value[0])}
                          />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* 翻译相关操作(语言选、字幕开关、开始翻译/下载) */}
                  <div className="flex items-center">
                    <TranslationControls
                      state={translationState}
                      controls={translationControls}
                    />
                  </div>

                  <Button 
                    size="icon" 
                    variant="ghost" 
                    className="hover:bg-white/10 active:scale-95 transition-transform text-white/70 hover:text-white h-7 w-7"
                    onClick={toggleFullscreen}
                  >
                    {isFullscreen ? (
                      <Minimize className="h-3 w-3" />
                    ) : (
                      <Maximize className="h-3 w-3" />
                    )}
                  </Button>
                </div>
              </div>
            </div>
            {/* 底部控制栏结束 */}
          </div>
        </div>
      </div>
    </div>
  )
}