import { useRef, useEffect } from 'react'
import Hls from 'hls.js'
import { toast } from 'sonner'
import { API_BASE_URL } from '../utils/format'
import type { HLSInstance } from '../types'

export function useHLSPlayer(
  videoRef: React.RefObject<HTMLVideoElement>,
  isPlaying: boolean
): HLSInstance {
  const hlsRef = useRef<Hls | null>(null)
  const retryCountRef = useRef<number>(0)
  const maxRetries = 3
  const retryDelayMs = 2000
  const currentTaskIdRef = useRef<string | null>(null)

  const initHLS = (taskId: string) => {
    if (!videoRef.current) return
    
    // 防止重复初始化相同的taskId
    if (hlsRef.current && currentTaskIdRef.current === taskId) {
      console.log(`HLS播放器已经初始化，taskId: ${taskId}，跳过重复初始化`)
      return
    }
    
    currentTaskIdRef.current = taskId
    
    const video = videoRef.current
    const playlistUrl = `${API_BASE_URL}/playlists/${taskId}/playlist_${taskId}.m3u8`

    // 清理之前的 HLS 实例
    if (hlsRef.current) {
      hlsRef.current.destroy()
    }

    console.log(`初始化HLS播放器，播放列表: ${playlistUrl}`)
    retryCountRef.current = 0

    if (Hls.isSupported()) {
      // 使用简化的HLS配置，与旧版本保持一致
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
        console.log(`HLS播放列表解析成功: ${playlistUrl}`)
        if (isPlaying) {
          video.play().catch(error => {
            console.log('自动播放失败:', error)
          })
        }
      })

      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          switch(data.type) {
            case Hls.ErrorTypes.NETWORK_ERROR:
              console.log(`网络错误，尝试恢复... URL: ${playlistUrl}, 详情:`, data.details)
              
              if (data.details === Hls.ErrorDetails.MANIFEST_LOAD_ERROR ||
                  data.details === Hls.ErrorDetails.MANIFEST_LOAD_TIMEOUT) {
                
                if (retryCountRef.current < maxRetries) {
                  retryCountRef.current++
                  console.log(`播放列表加载失败，${retryDelayMs/1000}秒后进行第${retryCountRef.current}次重试`)
                  
                  setTimeout(() => {
                    console.log(`正在重试加载播放列表: ${playlistUrl}`)
                    hls.loadSource(playlistUrl)
                    hls.startLoad()
                  }, retryDelayMs)
                } else {
                  console.error(`播放列表加载失败，已达到最大重试次数(${maxRetries})`)
                  toast.error('视频播放列表加载失败，可能是处理尚未完成，请稍后再试')
                }
                return
              }
              
              hls.startLoad()
              break
            case Hls.ErrorTypes.MEDIA_ERROR:
              console.log(`媒体错误，尝试恢复... 详情:`, data.details)
              hls.recoverMediaError()
              break
            default:
              console.error(`无法恢复的错误: 类型=${data.type}, 详情=${data.details}, URL=${playlistUrl}`)
              toast.error('视频播放出错，请稍后重试。')
              break
          }
        } else {
          console.warn(`HLS非致命错误: 类型=${data.type}, 详情=${data.details}`)
        }
      })
    } else {
      console.log(`浏览器不支持HLS.js，尝试使用原生HLS支持: ${playlistUrl}`)
      video.src = playlistUrl
      video.addEventListener('loadedmetadata', () => {
        if (isPlaying) {
          video.play().catch(error => {
            console.log('自动播放失败:', error)
          })
        }
      })
    }
  }

  const destroyHLS = () => {
    if (hlsRef.current) {
      console.log('销毁HLS播放器实例')
      hlsRef.current.destroy()
      hlsRef.current = null
      retryCountRef.current = 0
      currentTaskIdRef.current = null
    }
  }

  useEffect(() => {
    return () => {
      destroyHLS()
    }
  }, [])

  return {
    initHLS,
    destroyHLS
  }
} 