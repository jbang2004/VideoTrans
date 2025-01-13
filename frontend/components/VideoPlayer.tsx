// frontend/components/VideoPlayer.tsx
"use client"

import { useEffect, useRef } from 'react'
import Hls from 'hls.js'
import { toast } from "sonner"

interface VideoPlayerProps {
  taskId: string
  apiBaseUrl: string
}

export default function VideoPlayer({ taskId, apiBaseUrl }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const hlsRef = useRef<Hls | null>(null)

  useEffect(() => {
    if (!videoRef.current) return

    const video = videoRef.current
    const playlistUrl = `${apiBaseUrl}/playlists/playlist_${taskId}.m3u8`

    // 清理之前的 HLS 实例
    if (hlsRef.current) {
      hlsRef.current.destroy()
    }

    if (Hls.isSupported()) {
      const hls = new Hls({
        debug: false,
        enableWorker: true,
        maxBufferSize: 0,
        maxBufferLength: 30,
        manifestLoadingTimeOut: 20000,
        manifestLoadingMaxRetry: 3,
        levelLoadingTimeOut: 20000,
        fragLoadingTimeOut: 20000
      })

      hls.loadSource(playlistUrl)
      hls.attachMedia(video)
      hlsRef.current = hls

      hls.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play().catch(error => {
          console.log('自动播放失败:', error)
        })
      })

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          switch(data.type) {
            case Hls.ErrorTypes.NETWORK_ERROR:
              console.log('网络错误，尝试恢复...')
              hls.startLoad()
              break
            case Hls.ErrorTypes.MEDIA_ERROR:
              console.log('媒体错误，尝试恢复...')
              hls.recoverMediaError()
              break
            default:
              console.error('无法恢复的错误:', data)
              toast.error('视频播放出错，请稍后重试。')
              break
          }
        }
      })
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      video.src = playlistUrl
      video.addEventListener('loadedmetadata', () => {
        video.play().catch(error => {
          console.log('自动播放失败:', error)
        })
      })
    }

    return () => {
      if (hlsRef.current) {
        hlsRef.current.destroy()
      }
    }
  }, [taskId, apiBaseUrl])

  return (
    <div className="rounded-xl overflow-hidden bg-black aspect-video">
      <video
        ref={videoRef}
        className="w-full h-full"
        controls
        playsInline
      />
    </div>
  )
}